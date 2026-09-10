# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness-check Generator (tenstorrent/skills contract) for Kimi-Linear-48B-A3B-Instruct.

Single-user driver: owns the paged latent caches + KDA slot state, runs prefill eagerly and decode through a captured
TTNN trace (persistent device inputs refreshed with copy_host_to_device_tensor; all state updates are in-place).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
from loguru import logger

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.weights import KimiCheckpoint
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.layer import PrecisionPolicy
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.model import KimiLinearModel

for _p in (os.environ.get("RT"),):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from readiness_check.contract import Generator as _ContractGenerator
except Exception:  # the contract package is optional outside readiness runs

    class _ContractGenerator:  # type: ignore
        tokenizer: Any = None


def resolve_snapshot(model_dir: str | Path | None = None) -> Path:
    for key in ("KIMI_SNAPSHOT", "MODEL_WEIGHTS_DIR"):
        v = os.environ.get(key)
        if v and Path(v).is_dir():
            return Path(v)
    hf = os.environ.get("HF_MODEL", "moonshotai/Kimi-Linear-48B-A3B-Instruct")
    if Path(hf).is_dir():
        return Path(hf)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"))


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)


class KimiGenerator(_ContractGenerator):
    def __init__(
        self,
        mesh_device,
        *,
        snapshot: Path,
        max_seq_len: int = 8192,
        block_size: int = 64,
        precision: PrecisionPolicy | None = None,
        cache_path: Path | None = None,
        tokenizer=None,
    ):
        self.mesh_device = mesh_device
        self.cfg = KimiLinearConfig.from_snapshot(snapshot)
        self.cfg.validate()
        ck = KimiCheckpoint(snapshot, self.cfg)
        cache_path = cache_path or (
            Path(os.environ["TT_CACHE_PATH"]) / "kimi_linear_48b" / f"tp{tuple(mesh_device.shape)[1]}"
            if os.environ.get("TT_CACHE_PATH")
            else None
        )
        t0 = time.time()
        self.model = KimiLinearModel(
            mesh_device,
            self.cfg,
            ck,
            max_batch_size=1,
            cache_path=cache_path,
            precision=precision,
            block_size=block_size,
        )
        ck.close()
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.num_blocks = -(-max_seq_len // block_size) + 1
        self.model.allocate_state(self.num_blocks)
        self.page_table = torch.arange(self.num_blocks, dtype=torch.int32).reshape(1, -1)
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(snapshot)
        self.vocab = self.cfg.vocab_size
        self._trace = None
        logger.info(
            f"KimiGenerator ready in {time.time()-t0:.0f}s (max_seq_len {max_seq_len}, {self.num_blocks} blocks)"
        )

    # ---- low level -------------------------------------------------------------------------------
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kw,
    ):
        assert tokens.shape[0] == 1, "single-user generator"
        n = int(prompt_lens[0])
        pt = page_table if page_table is not None else self.page_table
        if return_all_logits:
            logits = self.model.prefill_all_logits(tokens[0, :n], pt[0:1], slot=0)  # [n, vocab]
            return logits.unsqueeze(0)
        last = self.model.prefill(tokens[0, :n], pt[0:1], slot=0)
        return last.reshape(1, 1, -1)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        **kw,
    ):
        pt = page_table if page_table is not None else self.page_table
        tok = tokens.reshape(-1)[:1]
        pos = start_pos.reshape(-1)[:1].to(torch.int32)
        if enable_trace:
            return self._decode_traced(tok, pos, pt).reshape(1, -1)
        return self.model.decode(tok, pos, pt).reshape(1, -1)

    # ---- high level ------------------------------------------------------------------------------
    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        self.reset()
        ids = torch.tensor(prompt_token_ids, dtype=torch.long)
        return self.model.prefill_all_logits(ids, self.page_table, slot=0).unsqueeze(0)  # [1, n, vocab]

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        stop_on_eos: bool = False,
        **kw,
    ) -> List[int]:
        self.reset()
        ids = torch.tensor(prompt_token_ids, dtype=torch.long)
        assert ids.numel() + max_new_tokens <= self.max_seq_len, (ids.numel(), max_new_tokens, self.max_seq_len)
        t0 = time.time()
        logits = self.model.prefill(ids, self.page_table, slot=0)
        self.last_prefill_s = time.time() - t0
        preds: List[int] = []
        pos = ids.numel()
        pred = int(logits.argmax())
        preds.append(pred)
        nxt = next_input(0, pred) if next_input else pred
        step_times = []
        for i in range(1, max_new_tokens):
            if stop_on_eos and pred == self.cfg.eos_token_id:
                break
            t1 = time.time()
            lg = self.decode_forward(
                torch.tensor([[nxt]]),
                torch.tensor([pos], dtype=torch.int32),
                page_table=self.page_table,
                kv_cache=None,
                enable_trace=enable_trace,
            )
            step_times.append(time.time() - t1)
            pred = int(lg.argmax())
            preds.append(pred)
            nxt = next_input(i, pred) if next_input else pred
            pos += 1
        self.last_decode_ms = 1000 * sum(step_times) / max(1, len(step_times))
        return preds

    def reset(self) -> None:
        self.model.reset_prefill_scratch()
        self.model.reset_slot(0)
        # the MLA caches are rewritten by the next prefill and positions beyond it are never attended -> no zeroing needed

    # ---- traced decode ---------------------------------------------------------------------------
    def warmup_decode_trace(self) -> None:
        """Compile + capture the decode trace on a throwaway token from a clean state, then reset. Real steps only replay."""
        if self._trace is not None:
            return
        m = self.model
        # prefill warm-up first: compiles the prefill programs (a compile after capture can clobber the trace) and touches
        # every lazily created buffer before the capture below
        t0 = time.time()
        warm = torch.full((64,), self.cfg.pad_token_id, dtype=torch.long)
        m.prefill(warm, self.page_table, slot=0)
        m.reset_slot(0)
        logger.info(f"prefill warm-up in {time.time()-t0:.1f}s")
        tok = torch.tensor([self.cfg.pad_token_id])
        pos = torch.tensor([0], dtype=torch.int32)
        dev = [ttnn.to_device(h, m.mesh_device) for h in m._host_decode_inputs(tok, pos, self.page_table)]
        t0 = time.time()
        out = m.decode_device(*dev)  # compile pass
        ttnn.deallocate(out)
        tid = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
        out = m.decode_device(*dev)
        ttnn.end_trace_capture(m.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(m.mesh_device)
        self._trace = (tid, dev, out)
        logger.info(f"decode trace captured in {time.time()-t0:.1f}s")
        self.reset()

    def _decode_traced(self, tok: torch.Tensor, pos: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        if self._trace is None:
            raise RuntimeError(
                "decode trace not captured: call warmup_decode_trace() before the first prompt (tracing is required by the readiness runner)"
            )
        m = self.model
        tid, dev, out = self._trace
        for h, d in zip(m._host_decode_inputs(tok, pos, pt), dev):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(m.mesh_device, tid, cq_id=0, blocking=False)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(-1)[: self.vocab]


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> KimiGenerator:
    snapshot = resolve_snapshot(model_dir)
    max_seq_len = int(kwargs.get("max_seq_len") or os.environ.get("KIMI_MAX_SEQ_LEN", 8192))
    precision = PrecisionPolicy()
    if os.environ.get("KIMI_PRECISION", "").lower() in ("bfp4", "bfp4_experts"):
        precision.experts = ttnn.bfloat4_b
    if os.environ.get("KIMI_KV_BFP8") == "1":
        precision.kv_cache = ttnn.bfloat8_b
    gen = KimiGenerator(mesh_device, snapshot=snapshot, max_seq_len=max_seq_len, precision=precision)
    if os.environ.get("KIMI_SKIP_TRACE_WARMUP") != "1":
        gen.warmup_decode_trace()
    return gen
