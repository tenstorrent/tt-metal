# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
ToneColorConverter - Main user-facing API for voice conversion.

This provides a high-level interface matching the original OpenVoice API,
making it easy to:
1. Extract speaker embeddings from reference audio
2. Convert voice from source audio using target speaker
3. Batch process multiple conversions efficiently
4. Pipeline extraction with synthesis for optimal throughput
"""

import hashlib
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Global model cache for reusing loaded models
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_cached_model(checkpoint_path: str) -> Optional[Any]:
    """Get a cached model if available."""
    with _MODEL_CACHE_LOCK:
        return _MODEL_CACHE.get(checkpoint_path)


def cache_model(checkpoint_path: str, model: Any):
    """Cache a loaded model for reuse."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[checkpoint_path] = model


def clear_model_cache():
    """Clear the global model cache."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class VoiceEmbeddingCache:
    """
    LRU cache for voice embeddings with optional disk persistence.

    Caches speaker embeddings to avoid recomputing them for frequently used voices.
    Supports both in-memory LRU caching and disk-based persistence.
    """

    def __init__(
        self,
        max_memory_entries: int = 100,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_disk_cache: bool = True,
    ):
        """
        Initialize the voice embedding cache.

        Args:
            max_memory_entries: Maximum number of embeddings to keep in memory
            cache_dir: Directory for disk cache (default: ~/.openvoice_cache)
            enable_disk_cache: Whether to persist embeddings to disk
        """
        self.max_memory_entries = max_memory_entries
        self.enable_disk_cache = enable_disk_cache

        # LRU cache using OrderedDict
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

        # Disk cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".openvoice_cache" / "embeddings"
        self.cache_dir = Path(cache_dir)
        if enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key from audio file path and modification time."""
        path = Path(audio_path)
        if path.exists():
            # Include file size and mtime for cache invalidation
            stat = path.stat()
            key_str = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        else:
            key_str = str(path.absolute())
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        Get embedding from cache.

        Args:
            audio_path: Path to audio file

        Returns:
            Cached embedding or None if not found
        """
        key = self._get_cache_key(audio_path)

        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                self.hits += 1
                return self._memory_cache[key].clone()

        # Check disk cache
        if self.enable_disk_cache:
            disk_path = self.cache_dir / f"{key}.pt"
            if disk_path.exists():
                try:
                    embedding = torch.load(disk_path, map_location="cpu")
                    # Add to memory cache
                    self._add_to_memory(key, embedding)
                    self.hits += 1
                    return embedding.clone()
                except Exception:
                    # Corrupted cache file, remove it
                    disk_path.unlink(missing_ok=True)

        self.misses += 1
        return None

    def put(self, audio_path: str, embedding: torch.Tensor):
        """
        Store embedding in cache.

        Args:
            audio_path: Path to audio file
            embedding: Speaker embedding tensor
        """
        key = self._get_cache_key(audio_path)

        # Add to memory cache
        self._add_to_memory(key, embedding.clone().cpu())

        # Save to disk cache
        if self.enable_disk_cache:
            disk_path = self.cache_dir / f"{key}.pt"
            try:
                torch.save(embedding.cpu(), disk_path)
            except Exception:
                pass  # Ignore disk write failures

    def _add_to_memory(self, key: str, embedding: torch.Tensor):
        """Add embedding to memory cache with LRU eviction."""
        with self._lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
            else:
                self._memory_cache[key] = embedding
                # Evict oldest if over capacity
                while len(self._memory_cache) > self.max_memory_entries:
                    self._memory_cache.popitem(last=False)

    def clear_memory(self):
        """Clear in-memory cache."""
        with self._lock:
            self._memory_cache.clear()

    def clear_disk(self):
        """Clear disk cache."""
        if self.enable_disk_cache and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pt"):
                f.unlink(missing_ok=True)

    def clear_all(self):
        """Clear both memory and disk caches."""
        self.clear_memory()
        self.clear_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "memory_entries": len(self._memory_cache),
            "max_memory_entries": self.max_memory_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "disk_cache_enabled": self.enable_disk_cache,
            "disk_cache_dir": str(self.cache_dir),
        }

    def preload(self, audio_paths: List[str], extract_fn: Callable[[str], torch.Tensor]):
        """
        Preload embeddings for a list of audio files.

        Args:
            audio_paths: List of audio file paths
            extract_fn: Function to extract embedding from audio path
        """
        for path in audio_paths:
            if self.get(path) is None:
                try:
                    embedding = extract_fn(path)
                    self.put(path, embedding)
                except Exception:
                    pass  # Skip failed extractions


try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.synthesizer import TTNNSynthesizerTrn
from models.demos.openvoice.utils.audio import AudioProcessor, save_audio
from models.demos.openvoice.utils.weight_loader import load_openvoice_checkpoint


@dataclass
class BatchConversionItem:
    """Single item in a batch conversion job."""

    source_audio: Union[str, Path]
    reference_audio: Union[str, List[str], Path]
    output_path: Union[str, Path]
    tau: float = 0.3
    source_se: Optional[torch.Tensor] = None
    # Results populated after conversion
    result_audio: Optional[np.ndarray] = None
    error: Optional[str] = None
    extraction_time: float = 0.0
    conversion_time: float = 0.0


@dataclass
class PipelineStats:
    """Statistics from pipelined processing."""

    total_items: int = 0
    successful: int = 0
    failed: int = 0
    total_extraction_time: float = 0.0
    total_conversion_time: float = 0.0
    wall_time: float = 0.0

    @property
    def throughput(self) -> float:
        """Items per second."""
        return self.total_items / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Average total latency per item."""
        return self.wall_time / self.total_items if self.total_items > 0 else 0.0


class TTNNToneColorConverter:
    """
    ToneColorConverter for voice cloning and conversion.

    High-level API matching the original OpenVoice ToneColorConverter.

    Usage:
        ```python
        # Initialize
        converter = TTNNToneColorConverter(config_path, device=device)
        converter.load_checkpoint(checkpoint_path)

        # Extract speaker embedding from reference
        target_se = converter.extract_se(["reference.wav"])

        # Convert voice
        converter.convert(
            source_audio="source.wav",
            src_se=source_se,
            tgt_se=target_se,
            output_path="converted.wav",
        )
        ```
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        device: Optional[Any] = None,
        enable_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        max_cache_entries: int = 100,
    ):
        """
        Initialize ToneColorConverter.

        Args:
            config_path: Path to config.json
            device: TTNN device (None for CPU mode)
            enable_cache: Enable voice embedding cache for faster repeated access
            cache_dir: Custom cache directory (default: ~/.openvoice_cache)
            max_cache_entries: Maximum embeddings to keep in memory
        """
        self.device = device
        self.config_path = Path(config_path)

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Get config values
        data_cfg = self.config.get("data", self.config)
        self.sample_rate = data_cfg.get("sampling_rate", 22050)

        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.config)

        # Model (loaded later)
        self.model: Optional[TTNNSynthesizerTrn] = None

        # Version info
        self.version = self.config.get("_version_", "v2")

        # Voice embedding cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.voice_cache = VoiceEmbeddingCache(
                max_memory_entries=max_cache_entries,
                cache_dir=cache_dir,
                enable_disk_cache=True,
            )
        else:
            self.voice_cache = None

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        use_model_cache: bool = True,
    ):
        """
        Load model checkpoint.

        Uses model cache to avoid reloading weights for the same checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.pth
            use_model_cache: Whether to use global model cache (default: True)
        """
        checkpoint_path = Path(checkpoint_path)
        cache_key = f"{checkpoint_path.absolute()}:{self.device}"

        # Check model cache first
        if use_model_cache:
            cached_model = get_cached_model(cache_key)
            if cached_model is not None:
                self.model = cached_model
                return

        weights, _ = load_openvoice_checkpoint(
            str(checkpoint_path),
            config_path=str(self.config_path),
            device=self.device,
        )

        self.model = TTNNSynthesizerTrn.from_state_dict(
            weights,
            self.config,
            device=self.device,
        )

        # Cache the model
        if use_model_cache:
            cache_model(cache_key, self.model)

    def extract_se(
        self,
        ref_wav_list: Union[str, List[str]],
        se_save_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Extract speaker embedding from reference audio(s).

        Uses voice embedding cache for faster repeated access to frequently
        used voices.

        Args:
            ref_wav_list: Single path or list of paths to reference audio
            se_save_path: Optional path to save the embedding
            use_cache: Whether to use the embedding cache (default: True)

        Returns:
            Speaker embedding tensor [1, gin_channels, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if self.model.ref_enc is None:
            raise RuntimeError("Model does not have reference encoder (n_speakers > 0)")

        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        embeddings = []

        for audio_path in ref_wav_list:
            # Check cache first
            if use_cache and self.voice_cache is not None:
                cached = self.voice_cache.get(audio_path)
                if cached is not None:
                    embeddings.append(cached)
                    continue

            # Load and process audio
            spec = self.audio_processor.load_and_process(audio_path)

            # ReferenceEncoder expects [B, T, n_freqs]
            # spec is [1, n_freqs, T], so transpose
            spec_input = spec.squeeze(0).T.unsqueeze(0)  # [1, T, n_freqs]

            # Move to device if needed
            if TTNN_AVAILABLE and self.device:
                spec_input = ttnn.from_torch(
                    spec_input.float(),
                    dtype=ttnn.bfloat16,
                    device=self.device,
                )

            # Extract embedding
            with torch.no_grad():
                g = self.model.ref_enc(spec_input)

            # Back to CPU if needed
            if TTNN_AVAILABLE and self.device and not isinstance(g, torch.Tensor):
                g = ttnn.to_torch(ttnn.from_device(g))

            # Add channel dimension [B, gin_channels] -> [B, gin_channels, 1]
            g = g.unsqueeze(-1).detach()

            # Store in cache
            if use_cache and self.voice_cache is not None:
                self.voice_cache.put(audio_path, g)

            embeddings.append(g)

        # Average embeddings if multiple references
        embedding = torch.stack(embeddings).mean(0)

        # Save if requested
        if se_save_path:
            save_dir = os.path.dirname(se_save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(embedding.cpu(), se_save_path)

        return embedding

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get voice embedding cache statistics."""
        if self.voice_cache is not None:
            return self.voice_cache.get_stats()
        return None

    def clear_cache(self, memory_only: bool = False):
        """
        Clear voice embedding cache.

        Args:
            memory_only: If True, only clear memory cache, keep disk cache
        """
        if self.voice_cache is not None:
            if memory_only:
                self.voice_cache.clear_memory()
            else:
                self.voice_cache.clear_all()

    def preload_voices(self, audio_paths: List[str]):
        """
        Preload voice embeddings into cache.

        Args:
            audio_paths: List of audio file paths to preload
        """
        if self.voice_cache is not None:
            self.voice_cache.preload(audio_paths, lambda p: self.extract_se([p], use_cache=False))

    def load_se(self, se_path: Union[str, Path]) -> torch.Tensor:
        """
        Load pre-computed speaker embedding.

        Args:
            se_path: Path to saved embedding

        Returns:
            Speaker embedding tensor
        """
        return torch.load(se_path, map_location="cpu")

    def convert(
        self,
        source_audio: Union[str, Path],
        src_se: torch.Tensor,
        tgt_se: torch.Tensor,
        output_path: Optional[Union[str, Path]] = None,
        tau: float = 0.3,
    ) -> np.ndarray:
        """
        Convert voice from source audio.

        Args:
            source_audio: Path to source audio file
            src_se: Source speaker embedding [1, gin_channels, 1]
            tgt_se: Target speaker embedding [1, gin_channels, 1]
            output_path: Optional path to save output (if None, returns array)
            tau: Temperature for conversion (lower = more similar to source)

        Returns:
            Converted audio as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        # Load and process source audio
        spec = self.audio_processor.load_and_process(source_audio)
        spec_lengths = torch.LongTensor([spec.size(-1)])

        # Move to device if needed
        if TTNN_AVAILABLE and self.device:
            spec = ttnn.from_torch(spec.float(), dtype=ttnn.bfloat16, device=self.device)
            spec_lengths = ttnn.from_torch(spec_lengths, dtype=ttnn.int32, device=self.device)
            src_se = ttnn.from_torch(src_se.float(), dtype=ttnn.bfloat16, device=self.device)
            tgt_se = ttnn.from_torch(tgt_se.float(), dtype=ttnn.bfloat16, device=self.device)

        # Run voice conversion
        with torch.no_grad():
            audio, _, _ = self.model.voice_conversion(spec, spec_lengths, src_se, tgt_se, tau=tau)

        # Back to CPU
        if TTNN_AVAILABLE and self.device and not isinstance(audio, torch.Tensor):
            audio = ttnn.to_torch(ttnn.from_device(audio))

        # Convert to numpy
        audio_np = audio[0, 0].cpu().float().numpy()

        # Save if requested
        if output_path:
            save_audio(output_path, audio_np, sr=self.sample_rate)

        return audio_np

    def convert_from_files(
        self,
        source_audio: Union[str, Path],
        reference_audio: Union[str, List[str], Path],
        output_path: Union[str, Path],
        tau: float = 0.3,
        source_se: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Convenience method: extract embeddings and convert in one call.

        Args:
            source_audio: Path to source audio
            reference_audio: Path(s) to reference audio for target voice
            output_path: Path to save converted audio
            tau: Temperature
            source_se: Optional pre-computed source embedding

        Returns:
            Converted audio array
        """
        # Extract target speaker embedding
        if isinstance(reference_audio, (str, Path)):
            reference_audio = [str(reference_audio)]
        tgt_se = self.extract_se(reference_audio)

        # Extract source speaker embedding if not provided
        if source_se is None:
            src_se = self.extract_se([str(source_audio)])
        else:
            src_se = source_se

        # Convert
        return self.convert(
            source_audio=source_audio,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=output_path,
            tau=tau,
        )

    # =========================================================================
    # BATCH PROCESSING METHODS
    # =========================================================================

    def extract_se_batch(
        self,
        audio_paths: List[Union[str, Path]],
        num_workers: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract speaker embeddings from multiple audio files in parallel.

        Uses CPU parallelism for audio loading and preprocessing, then
        batches GPU inference for efficiency.

        Args:
            audio_paths: List of paths to audio files
            num_workers: Number of parallel workers for preprocessing

        Returns:
            Dict mapping audio path to speaker embedding tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if self.model.ref_enc is None:
            raise RuntimeError("Model does not have reference encoder")

        results = {}
        audio_paths = [str(p) for p in audio_paths]

        # Step 1: Parallel preprocessing (CPU-bound)
        def preprocess_audio(audio_path: str) -> Tuple[str, torch.Tensor]:
            spec = self.audio_processor.load_and_process(audio_path)
            spec_input = spec.squeeze(0).T.unsqueeze(0)  # [1, T, n_freqs]
            return audio_path, spec_input

        preprocessed = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(preprocess_audio, p): p for p in audio_paths}
            for future in as_completed(futures):
                try:
                    path, spec = future.result()
                    preprocessed.append((path, spec))
                except Exception:
                    pass  # Skip failed preprocessing

        # Step 2: Batch inference on TTNN
        with torch.no_grad():
            for path, spec_input in preprocessed:
                try:
                    if TTNN_AVAILABLE and self.device:
                        spec_ttnn = ttnn.from_torch(
                            spec_input.float(),
                            dtype=ttnn.bfloat16,
                            device=self.device,
                        )
                        g = self.model.ref_enc(spec_ttnn)
                        if not isinstance(g, torch.Tensor):
                            g = ttnn.to_torch(ttnn.from_device(g))
                    else:
                        g = self.model.ref_enc(spec_input)

                    g = g.unsqueeze(-1).detach()
                    results[path] = g
                except Exception:
                    pass  # Skip failed extraction

        return results

    def convert_batch(
        self,
        items: List[BatchConversionItem],
        num_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BatchConversionItem]:
        """
        Convert multiple voice conversions in a batch.

        Optimizes throughput by:
        1. Parallel audio preprocessing on CPU
        2. Batched embedding extraction
        3. Sequential conversion on TTNN (maintains quality)

        Args:
            items: List of BatchConversionItem to process
            num_workers: Number of parallel workers for CPU tasks
            progress_callback: Optional callback(completed, total)

        Returns:
            List of BatchConversionItem with results populated
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        total = len(items)

        # Collect all unique audio files for embedding extraction
        all_audio_files = set()
        for item in items:
            all_audio_files.add(str(item.source_audio))
            if isinstance(item.reference_audio, (str, Path)):
                all_audio_files.add(str(item.reference_audio))
            else:
                all_audio_files.update([str(p) for p in item.reference_audio])

        # Extract all embeddings at once
        embeddings_cache = self.extract_se_batch(list(all_audio_files), num_workers)

        # Process each conversion
        completed = 0
        for item in items:
            try:
                start_time = time.time()

                # Get source embedding
                if item.source_se is not None:
                    src_se = item.source_se
                elif str(item.source_audio) in embeddings_cache:
                    src_se = embeddings_cache[str(item.source_audio)]
                else:
                    src_se = self.extract_se([str(item.source_audio)])
                item.extraction_time = time.time() - start_time

                # Get target embedding (average if multiple)
                ref_paths = item.reference_audio
                if isinstance(ref_paths, (str, Path)):
                    ref_paths = [str(ref_paths)]
                else:
                    ref_paths = [str(p) for p in ref_paths]

                tgt_embeddings = [embeddings_cache.get(p) for p in ref_paths if p in embeddings_cache]
                if tgt_embeddings:
                    tgt_se = torch.stack(tgt_embeddings).mean(0)
                else:
                    tgt_se = self.extract_se(ref_paths)

                # Run conversion
                conv_start = time.time()
                result = self.convert(
                    source_audio=item.source_audio,
                    src_se=src_se,
                    tgt_se=tgt_se,
                    output_path=item.output_path,
                    tau=item.tau,
                )
                item.result_audio = result
                item.conversion_time = time.time() - conv_start

            except Exception as e:
                item.error = str(e)

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

        successful = sum(1 for item in items if item.error is None)

        return items

    # =========================================================================
    # PIPELINING METHODS
    # =========================================================================

    def convert_pipelined(
        self,
        items: List[BatchConversionItem],
        num_workers: int = 4,
        queue_depth: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[BatchConversionItem], PipelineStats]:
        """
        Pipeline extraction with synthesis for maximum throughput.

        Overlaps CPU-bound preprocessing with TTNN inference:
        - Worker threads: Load audio, compute spectrograms
        - Main thread: Run TTNN inference while workers prepare next batch

        Args:
            items: List of BatchConversionItem to process
            num_workers: Parallel workers for preprocessing
            queue_depth: Number of items to keep ready in queue
            progress_callback: Optional callback(completed, total)

        Returns:
            Tuple of (processed items, pipeline statistics)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        stats = PipelineStats(total_items=len(items))
        start_wall_time = time.time()

        # Queue for preprocessed items ready for conversion
        ready_queue: queue.Queue = queue.Queue(maxsize=queue_depth)
        done_event = threading.Event()

        # Track preprocessing results
        prep_results: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        prep_errors: Dict[int, str] = {}

        def preprocess_item(idx: int, item: BatchConversionItem):
            """Preprocess a single item: load audio, extract embeddings."""
            try:
                extract_start = time.time()

                # Load source spectrogram
                spec = self.audio_processor.load_and_process(str(item.source_audio))
                spec_input = spec.squeeze(0).T.unsqueeze(0)

                # Get source embedding
                if item.source_se is not None:
                    src_se = item.source_se
                else:
                    with torch.no_grad():
                        if TTNN_AVAILABLE and self.device:
                            spec_ttnn = ttnn.from_torch(
                                spec_input.float(),
                                dtype=ttnn.bfloat16,
                                device=self.device,
                            )
                            src_se = self.model.ref_enc(spec_ttnn)
                            if not isinstance(src_se, torch.Tensor):
                                src_se = ttnn.to_torch(ttnn.from_device(src_se))
                        else:
                            src_se = self.model.ref_enc(spec_input)
                        src_se = src_se.unsqueeze(-1).detach()

                # Get target embedding
                ref_paths = item.reference_audio
                if isinstance(ref_paths, (str, Path)):
                    ref_paths = [str(ref_paths)]
                else:
                    ref_paths = [str(p) for p in ref_paths]

                tgt_embeddings = []
                for ref_path in ref_paths:
                    ref_spec = self.audio_processor.load_and_process(ref_path)
                    ref_input = ref_spec.squeeze(0).T.unsqueeze(0)
                    with torch.no_grad():
                        if TTNN_AVAILABLE and self.device:
                            ref_ttnn = ttnn.from_torch(
                                ref_input.float(),
                                dtype=ttnn.bfloat16,
                                device=self.device,
                            )
                            tgt_se = self.model.ref_enc(ref_ttnn)
                            if not isinstance(tgt_se, torch.Tensor):
                                tgt_se = ttnn.to_torch(ttnn.from_device(tgt_se))
                        else:
                            tgt_se = self.model.ref_enc(ref_input)
                        tgt_embeddings.append(tgt_se.unsqueeze(-1).detach())

                tgt_se = torch.stack(tgt_embeddings).mean(0)

                extract_time = time.time() - extract_start
                items[idx].extraction_time = extract_time

                return (idx, spec, src_se, tgt_se)

            except Exception as e:
                return (idx, None, None, str(e))

        def preprocessing_worker():
            """Worker thread that preprocesses items and puts them in queue."""
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(preprocess_item, i, item): i for i, item in enumerate(items)}

                for future in as_completed(futures):
                    if done_event.is_set():
                        break
                    result = future.result()
                    idx = result[0]

                    if result[2] is None:  # Error case
                        prep_errors[idx] = result[3]
                    else:
                        ready_queue.put(result)

            # Signal completion
            ready_queue.put(None)

        # Start preprocessing thread
        prep_thread = threading.Thread(target=preprocessing_worker)
        prep_thread.start()

        # Main thread: consume from queue and run conversions
        completed = 0
        processed_indices = set()

        try:
            while True:
                prep_result = ready_queue.get(timeout=60)
                if prep_result is None:
                    break

                idx, spec, src_se, tgt_se = prep_result

                try:
                    conv_start = time.time()

                    # Move tensors to device
                    spec_lengths = torch.LongTensor([spec.size(-1)])

                    if TTNN_AVAILABLE and self.device:
                        spec = ttnn.from_torch(spec.float(), dtype=ttnn.bfloat16, device=self.device)
                        spec_lengths = ttnn.from_torch(spec_lengths, dtype=ttnn.int32, device=self.device)
                        src_se = ttnn.from_torch(src_se.float(), dtype=ttnn.bfloat16, device=self.device)
                        tgt_se = ttnn.from_torch(tgt_se.float(), dtype=ttnn.bfloat16, device=self.device)

                    # Run conversion
                    with torch.no_grad():
                        audio, _, _ = self.model.voice_conversion(
                            spec, spec_lengths, src_se, tgt_se, tau=items[idx].tau
                        )

                    # Back to CPU
                    if TTNN_AVAILABLE and self.device and not isinstance(audio, torch.Tensor):
                        audio = ttnn.to_torch(ttnn.from_device(audio))

                    audio_np = audio[0, 0].cpu().float().numpy()

                    # Save output
                    if items[idx].output_path:
                        save_audio(items[idx].output_path, audio_np, sr=self.sample_rate)

                    items[idx].result_audio = audio_np
                    items[idx].conversion_time = time.time() - conv_start
                    stats.successful += 1

                except Exception as e:
                    items[idx].error = str(e)
                    stats.failed += 1

                processed_indices.add(idx)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

        except queue.Empty:
            pass  # Pipeline timeout, continue to cleanup
        finally:
            done_event.set()
            prep_thread.join(timeout=5)

        # Handle any preprocessing errors
        for idx, error in prep_errors.items():
            if idx not in processed_indices:
                items[idx].error = f"Preprocessing failed: {error}"
                stats.failed += 1

        stats.wall_time = time.time() - start_wall_time
        stats.total_extraction_time = sum(item.extraction_time for item in items)
        stats.total_conversion_time = sum(item.conversion_time for item in items)

        return items, stats

    def convert_batch_simple(
        self,
        source_audios: List[Union[str, Path]],
        reference_audio: Union[str, Path],
        output_dir: Union[str, Path],
        tau: float = 0.3,
        use_pipeline: bool = True,
    ) -> Tuple[List[np.ndarray], PipelineStats]:
        """
        Simple batch API: convert multiple source audios to same target voice.

        Args:
            source_audios: List of source audio paths
            reference_audio: Single reference audio for target voice
            output_dir: Directory to save outputs
            tau: Temperature parameter
            use_pipeline: Use pipelining for better throughput

        Returns:
            Tuple of (list of converted audio arrays, statistics)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create batch items
        items = []
        for i, src in enumerate(source_audios):
            src_path = Path(src)
            output_path = output_dir / f"converted_{i:04d}_{src_path.stem}.wav"
            items.append(
                BatchConversionItem(
                    source_audio=src,
                    reference_audio=reference_audio,
                    output_path=str(output_path),
                    tau=tau,
                )
            )

        # Process
        if use_pipeline:
            items, stats = self.convert_pipelined(items)
        else:
            items = self.convert_batch(items)
            stats = PipelineStats(
                total_items=len(items),
                successful=sum(1 for i in items if i.error is None),
                failed=sum(1 for i in items if i.error is not None),
            )

        results = [item.result_audio for item in items if item.result_audio is not None]
        return results, stats


def create_converter(
    checkpoint_dir: Union[str, Path],
    device: Optional[Any] = None,
) -> TTNNToneColorConverter:
    """
    Convenience function to create and load a ToneColorConverter.

    Args:
        checkpoint_dir: Directory containing config.json and checkpoint.pth
        device: TTNN device

    Returns:
        Loaded ToneColorConverter
    """
    checkpoint_dir = Path(checkpoint_dir)

    config_path = checkpoint_dir / "config.json"
    checkpoint_path = checkpoint_dir / "checkpoint.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    converter = TTNNToneColorConverter(config_path, device=device)
    converter.load_checkpoint(checkpoint_path)

    return converter
