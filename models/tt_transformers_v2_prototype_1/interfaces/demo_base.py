# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Base class for standardized demos"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DemoBase(ABC):
    """
    Abstract base class for TTT demos.

    Provides a standardized interface for running demos with consistent
    initialization, execution, and cleanup patterns.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        device_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.device_config = device_config or self.get_default_device_config()
        self.output_dir = Path(output_dir) if output_dir else Path("demo_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.metrics = {
            "model_name": model_name,
            "model_version": model_version,
            "device_config": device_config,
            "results": [],
        }

        # Initialize demo
        self._setup()

    @abstractmethod
    def _setup(self):
        """Setup demo resources (model, tokenizer, etc.)"""

    @abstractmethod
    def run_single(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Run demo on a single input.

        Args:
            input_text: Input text for the demo
            **kwargs: Additional demo-specific arguments

        Returns:
            Dictionary with demo results
        """

    @abstractmethod
    def get_default_device_config(self) -> Dict[str, Any]:
        """Get default device configuration for this demo"""

    def run_batch(self, input_texts: List[str], batch_size: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Run demo on multiple inputs.

        Args:
            input_texts: List of input texts
            batch_size: Optional batch size for processing
            **kwargs: Additional demo-specific arguments

        Returns:
            List of result dictionaries
        """
        results = []

        if batch_size is None:
            # Process one by one
            for text in input_texts:
                result = self.run_single(text, **kwargs)
                results.append(result)
        else:
            # Process in batches
            for i in range(0, len(input_texts), batch_size):
                batch = input_texts[i : i + batch_size]
                batch_results = self._run_batch_internal(batch, **kwargs)
                results.extend(batch_results)

        return results

    def _run_batch_internal(self, batch: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Internal method for batch processing.

        Default implementation processes sequentially.
        Override for true batch processing.
        """
        return [self.run_single(text, **kwargs) for text in batch]

    def benchmark(
        self, input_texts: List[str], warmup_runs: int = 3, benchmark_runs: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark the demo performance.

        Args:
            input_texts: List of input texts for benchmarking
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            **kwargs: Additional demo-specific arguments

        Returns:
            Benchmark results including timing statistics
        """
        print(f"Running benchmark for {self.model_name} v{self.model_version}")
        print(f"Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")

        # Warmup
        print("Warming up...")
        for _ in range(warmup_runs):
            self.run_batch(input_texts, **kwargs)

        # Benchmark
        print("Benchmarking...")
        times = []
        for run in range(benchmark_runs):
            start_time = time.time()
            results = self.run_batch(input_texts, **kwargs)
            end_time = time.time()

            run_time = end_time - start_time
            times.append(run_time)

            print(f"Run {run + 1}/{benchmark_runs}: {run_time:.3f}s")

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = len(input_texts) / avg_time

        benchmark_results = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "num_inputs": len(input_texts),
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "times": times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": throughput,
            "device_config": self.device_config,
        }

        # Save results
        self._save_benchmark_results(benchmark_results)

        return benchmark_results

    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.model_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results saved to: {filepath}")

    def save_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Save demo results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results_{self.model_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        output_data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device_config": self.device_config,
            "timestamp": timestamp,
            "results": results,
        }

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {filepath}")

    def cleanup(self):
        """Cleanup demo resources"""
        # Override to add specific cleanup logic

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class TextGenerationDemo(DemoBase):
    """
    Base class for text generation demos.

    Provides common functionality for LLM demos.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        super().__init__(model_name, model_version, device_config, output_dir)

    def run_single(
        self, input_text: str, max_length: Optional[int] = None, temperature: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Run text generation on a single input.

        Args:
            input_text: Input prompt
            max_length: Override default max length
            temperature: Override default temperature
            **kwargs: Additional generation arguments

        Returns:
            Dictionary with generated text and metadata
        """
        # Use provided values or defaults
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature

        # Time the generation
        start_time = time.time()
        generated_text = self._generate(input_text, max_length=max_length, temperature=temperature, **kwargs)
        end_time = time.time()

        # Calculate tokens per second
        # Note: This is approximate without actual token counting
        generation_time = end_time - start_time
        approx_tokens = len(generated_text.split())
        tokens_per_second = approx_tokens / generation_time if generation_time > 0 else 0

        result = {
            "input": input_text,
            "output": generated_text,
            "generation_time": generation_time,
            "approx_tokens": approx_tokens,
            "tokens_per_second": tokens_per_second,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        }

        return result

    @abstractmethod
    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Internal generation method to be implemented by subclasses.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """


class MultimodalDemo(DemoBase):
    """
    Base class for multimodal demos.

    Handles demos that process both text and images.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        device_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ):
        super().__init__(model_name, model_version, device_config, output_dir)

    def run_single(self, input_text: str, image_path: Optional[Union[str, Path]] = None, **kwargs) -> Dict[str, Any]:
        """
        Run multimodal demo on text and optional image.

        Args:
            input_text: Input text/prompt
            image_path: Optional path to input image
            **kwargs: Additional demo-specific arguments

        Returns:
            Dictionary with demo results
        """
        # Load image if provided
        image = None
        if image_path:
            image = self._load_image(image_path)

        # Time the processing
        start_time = time.time()
        output = self._process_multimodal(text=input_text, image=image, **kwargs)
        end_time = time.time()

        result = {
            "input_text": input_text,
            "input_image": str(image_path) if image_path else None,
            "output": output,
            "processing_time": end_time - start_time,
        }

        return result

    @abstractmethod
    def _load_image(self, image_path: Union[str, Path]) -> Any:
        """Load and preprocess image"""

    @abstractmethod
    def _process_multimodal(self, text: str, image: Optional[Any], **kwargs) -> Any:
        """Process multimodal inputs"""
