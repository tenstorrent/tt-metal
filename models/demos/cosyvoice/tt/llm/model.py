"""CosyVoice LLM — TTNN wrapper around tt_transformers Qwen2.5-0.5B.

Reuses the tt_transformers Transformer (24× TransformerBlock, GQA attention,
SwiGLU MLP, RMSNorm, RoPE, KV cache) and adds CosyVoice-specific glue:
  - Two-table embedding assembler (Qwen embed_tokens + speech_embedding + sos/task)
  - llm_decoder head (896→6564) replacing the Qwen LMHead (896→151936)
  - Host-side RAS sampling
  - Custom autoregressive decode loop (prefill prefix → decode speech tokens)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger

import ttnn
from models.demos.cosyvoice.tt.llm.sampling import sampling_ids
from models.demos.cosyvoice.tt.model_config import LLM, SEED, SPEECH_TOKEN_VOCAB
from models.demos.cosyvoice.tt.weights import build_cosyvoice_state_dict
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

BLANKEN_PATH = str(Path(__file__).resolve().parents[2] / "model_data" / "cosyvoice2-0.5B" / "CosyVoice-BlankEN")
LLM_PT_PATH = str(Path(__file__).resolve().parents[2] / "model_data" / "cosyvoice2-0.5B" / "llm.pt")


def _compute_padded_vocab(vocab_size: int, tile_size: int = 32) -> int:
    return math.ceil(vocab_size / tile_size) * tile_size


class CosyVoiceLLM:
    """Stage-1 CosyVoice2 LLM on TTNN (single-device, batch=1, non-streaming)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        llm_pt_path: str = LLM_PT_PATH,
        blanken_path: str = BLANKEN_PATH,
        max_seq_len: int = 2048,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.speech_token_size = LLM.speech_embedding_num - 3  # 6561
        self.stop_token_ids = [self.speech_token_size + i for i in range(3)]
        self.sos = 0
        self.task_id = 1

        os.environ["HF_MODEL"] = blanken_path

        optimizations = DecodersPrecision.accuracy(num_decoders=LLM.num_hidden_layers, model_name="Qwen2.5-0.5B")

        self.args = ModelArgs(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            use_hf_rope=False,
        )

        self.args.vocab_size = SPEECH_TOKEN_VOCAB
        self.args.padded_vocab_size = _compute_padded_vocab(SPEECH_TOKEN_VOCAB)

        backbone_sd, speech_heads = build_cosyvoice_state_dict(llm_pt_path)

        self.model = Transformer(
            args=self.args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=backbone_sd,
            weight_cache_path=self.args.weight_cache_path(dtype),
        )

        self._setup_speech_embedding(speech_heads)
        self._trace_id = None

    def _setup_speech_embedding(self, speech_heads: Dict[str, torch.Tensor]):
        speech_emb_weight = speech_heads["speech_embedding.weight"].unsqueeze(0).unsqueeze(0)
        self.speech_embedding_weights = ttnn.as_tensor(
            speech_emb_weight,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        llm_emb = speech_heads["llm_embedding.weight"]
        self.sos_vec = llm_emb[self.sos]
        self.task_id_vec = llm_emb[self.task_id]

        self.text_embedding_weight = None

    def _get_text_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up Qwen text embeddings on host (from tok_embeddings.weight)."""
        if self.text_embedding_weight is None:
            sd = self.model.embd.weights
            self.text_embedding_weight = ttnn.to_torch(ttnn.get_device_tensors(sd)[0]).squeeze(0).squeeze(0)
        return self.text_embedding_weight[token_ids]

    def _get_speech_embeddings_host(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up speech embeddings on host."""
        sd = self.speech_embedding_weights
        host_weight = ttnn.to_torch(ttnn.get_device_tensors(sd)[0]).squeeze(0).squeeze(0)
        return host_weight[token_ids]

    def assemble_prefix(
        self,
        text_token_ids: torch.Tensor,
        prompt_speech_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Assemble the prefix embedding sequence on host.

        Returns: [1, T_prefix, 896] float32 tensor.
        Layout: [sos, text_emb..., task_id, prompt_speech_emb...]
        """
        text_emb = self._get_text_embeddings(text_token_ids)
        sos_emb = self.sos_vec.unsqueeze(0)
        task_emb = self.task_id_vec.unsqueeze(0)

        parts = [sos_emb, text_emb, task_emb]

        if prompt_speech_token_ids is not None and prompt_speech_token_ids.numel() > 0:
            prompt_speech_emb = self._get_speech_embeddings_host(prompt_speech_token_ids)
            parts.append(prompt_speech_emb)

        prefix = torch.cat(parts, dim=0).unsqueeze(0)
        return prefix

    def prefill(
        self,
        prefix_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefill on the assembled prefix embeddings.

        Args:
            prefix_embeds: [1, T_prefix, 896] host tensor

        Returns:
            logits: [SPEECH_TOKEN_VOCAB] host tensor (log-softmax applied)
        """
        seq_len = prefix_embeds.shape[1]
        padded_len = math.ceil(seq_len / 128) * 128

        if padded_len > seq_len:
            pad = torch.zeros(1, padded_len - seq_len, 896, dtype=prefix_embeds.dtype)
            prefix_embeds = torch.cat([prefix_embeds, pad], dim=1)

        tt_input = ttnn.from_torch(
            prefix_embeds.unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos_slice = self.model.rope_setup.cos_matrix_prefill[:, :, :padded_len, :]
        sin_slice = self.model.rope_setup.sin_matrix_prefill[:, :, :padded_len, :]
        rot_mats = [cos_slice, sin_slice]

        get_last_token = ((seq_len - 1) // 32) * 32

        tt_logits = self.model.ttnn_prefill_forward(
            tt_input,
            rot_mats_global=rot_mats,
            get_last_token=get_last_token,
            batch_size=1,
        )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits_torch = ttnn.to_torch(tt_logits)
        logits = logits_torch[0, 0, seq_len - 1 - get_last_token, :SPEECH_TOKEN_VOCAB].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

    def decode_step(
        self,
        speech_token_id: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Run a single decode step.

        Args:
            speech_token_id: the speech token to embed and feed
            current_pos: position in the sequence (for RoPE + KV cache)

        Returns:
            log_probs: [SPEECH_TOKEN_VOCAB] host tensor
        """
        if self._trace_id is not None:
            return self._decode_step_traced(speech_token_id, current_pos)

        token_tensor = ttnn.from_torch(
            torch.tensor([[speech_token_id]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        speech_emb = ttnn.embedding(
            token_tensor,
            self.speech_embedding_weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        speech_emb = ttnn.unsqueeze_to_4D(speech_emb)

        decode_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        speech_emb = ttnn.to_memory_config(speech_emb, decode_mem_cfg)

        position_idxs = torch.tensor([current_pos], dtype=torch.long)
        rot_mats = self.model.rope_setup.get_rot_mats(position_idxs)

        current_pos_tt = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tt_logits = self.model.forward(
            speech_emb,
            current_pos_tt,
            rot_mats_global=rot_mats,
            mode=Mode.DECODE,
        )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits_torch = ttnn.to_torch(tt_logits)
        logits = logits_torch[0, 0, 0, :SPEECH_TOKEN_VOCAB].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

    def _init_trace(self, start_pos: int):
        """Initialize trace capture for decode loop.

        Must be called after prefill, with the position of the first decode step.
        Performs a compile warmup, then captures the decode forward as a trace.
        """
        decode_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)

        self._trace_token_host = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._trace_pos_host = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
        )

        self._trace_token_buf = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._trace_pos_buf = ttnn.from_torch(
            torch.tensor([start_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        rot_idxs = self.model.rope_setup.get_rot_idxs(torch.tensor([start_pos]))
        self._trace_rot_idx_buf = rot_idxs
        self._trace_rot_idx_shape = rot_idxs.shape

        ttnn.copy_host_to_device_tensor(self._trace_token_host, self._trace_token_buf)
        ttnn.copy_host_to_device_tensor(self._trace_pos_host, self._trace_pos_buf)

        self._decode_forward_device()

        ttnn.copy_host_to_device_tensor(self._trace_token_host, self._trace_token_buf)
        ttnn.copy_host_to_device_tensor(self._trace_pos_host, self._trace_pos_buf)

        self._trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_output = self._decode_forward_device()
        ttnn.end_trace_capture(self.mesh_device, self._trace_id, cq_id=0)

        logger.info(f"Decode trace captured (start_pos={start_pos})")

    def _decode_forward_device(self) -> ttnn.Tensor:
        """Device-side decode forward (traceable). Reads from persistent buffers."""
        speech_emb = ttnn.embedding(
            self._trace_token_buf,
            self.speech_embedding_weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        speech_emb = ttnn.unsqueeze_to_4D(speech_emb)

        decode_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        speech_emb = ttnn.to_memory_config(speech_emb, decode_mem_cfg)

        rot_mats = self.model.rope_setup.get_rot_mats(self._trace_rot_idx_buf)

        tt_logits = self.model.forward(
            speech_emb,
            self._trace_pos_buf,
            rot_mats_global=rot_mats,
            mode=Mode.DECODE,
        )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_logits

    def _decode_step_traced(self, speech_token_id: int, current_pos: int) -> torch.Tensor:
        """Replay the captured decode trace with updated inputs."""
        token_host = ttnn.from_torch(
            torch.tensor([[speech_token_id]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pos_host = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
        )
        rot_idx_host = self.model.rope_setup.get_rot_idxs(torch.tensor([current_pos]), on_host=True)

        ttnn.copy_host_to_device_tensor(token_host, self._trace_token_buf)
        ttnn.copy_host_to_device_tensor(pos_host, self._trace_pos_buf)
        ttnn.copy_host_to_device_tensor(rot_idx_host, self._trace_rot_idx_buf)

        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)

        logits_torch = ttnn.to_torch(self._trace_output)
        logits = logits_torch[0, 0, 0, :SPEECH_TOKEN_VOCAB].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs

    def release_trace(self):
        """Release the captured decode trace."""
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
            self._trace_id = None

    @torch.inference_mode()
    def generate(
        self,
        text_token_ids: torch.Tensor,
        prompt_speech_token_ids: Optional[torch.Tensor] = None,
        min_len: int = 10,
        max_len: int = 500,
        top_p: float = 0.8,
        top_k: int = 25,
        win_size: int = 10,
        tau_r: float = 0.1,
        seed: int = SEED,
    ) -> List[int]:
        """Full autoregressive generation loop.

        Args:
            text_token_ids: [T_text] token IDs (prompt_text + text concatenated)
            prompt_speech_token_ids: [T_prompt_speech] speech token IDs (or None)
            min_len: minimum tokens before allowing EOS
            max_len: maximum generation length
            seed: RNG seed for RAS sampling

        Returns:
            List of generated speech token IDs (excluding EOS).
        """
        torch.manual_seed(seed)

        prefix = self.assemble_prefix(text_token_ids, prompt_speech_token_ids)
        log_probs = self.prefill(prefix)

        out_tokens: List[int] = []
        ignore_eos = True

        token_id = sampling_ids(
            log_probs,
            out_tokens,
            self.speech_token_size,
            ignore_eos=ignore_eos,
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )

        if token_id in self.stop_token_ids:
            return out_tokens

        out_tokens.append(token_id)
        current_pos = prefix.shape[1]

        if self._trace_id is None:
            self._init_trace(current_pos)

        for step in range(1, max_len):
            ignore_eos = step < min_len
            log_probs = self.decode_step(token_id, current_pos)
            current_pos += 1

            token_id = sampling_ids(
                log_probs,
                out_tokens,
                self.speech_token_size,
                ignore_eos=ignore_eos,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
            )

            if token_id in self.stop_token_ids:
                break

            out_tokens.append(token_id)

        return out_tokens

    @torch.inference_mode()
    def teacher_forced_step(
        self,
        lm_input: torch.Tensor,
        golden_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run teacher-forced decode: prefill prefix, then decode with golden tokens.

        Args:
            lm_input: [1, T_prefix, 896] — prefix embedding from golden fixture
            golden_tokens: [N] — golden token sequence

        Returns:
            log_probs: [N, SPEECH_TOKEN_VOCAB] — log-softmax at each decode step
        """
        log_probs = self.prefill(lm_input)
        all_log_probs = [log_probs]

        current_pos = lm_input.shape[1]
        for i in range(len(golden_tokens) - 1):
            token_id = golden_tokens[i].item()
            lp = self.decode_step(token_id, current_pos)
            all_log_probs.append(lp)
            current_pos += 1

        return torch.stack(all_log_probs, dim=0)
