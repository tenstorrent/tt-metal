# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check that a decode / MTP record matches ``lmsys/DeepSeek-R1-NextN`` config dimensions.

The NextN export is a **full** DeepSeek-V3-style decoder layer on disk (MoE + MLA), but the
**residual stream** width and vocabulary must match the base R1 record:

- ``hidden_states`` / ``step_last_hidden`` last dim == ``config.hidden_size`` (7168 for R1-class).
- All token ids in the record must be in ``[0, vocab_size)`` for NextN's ``vocab_size`` (129280).

This module only uses ``AutoConfig`` + the record file (no NextN weight download).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig

from models.demos.speculative_deepseek_r1_broad.trace_replay_base import (
    _is_mtp_reference_payload,
    load_mtp_reference_bundle,
    trace_bundle_from_collect_payload,
)


@dataclass(frozen=True)
class RecordNextNAlignmentReport:
    record_path: str
    nextn_model_id: str
    record_hidden_size: int
    record_max_token_id: int
    record_metadata_vocab_size: int | None
    nextn_hidden_size: int
    nextn_vocab_size: int
    hidden_matches: bool
    tokens_within_vocab: bool
    metadata_vocab_matches: bool | None

    @property
    def ok(self) -> bool:
        meta_ok = self.metadata_vocab_matches is not False
        return self.hidden_matches and self.tokens_within_vocab and meta_ok

    def summary_lines(self) -> list[str]:
        lines = [
            f"Record: {self.record_path}",
            f"NextN config repo: {self.nextn_model_id}",
            f"  record hidden_size: {self.record_hidden_size}",
            f"  NextN hidden_size: {self.nextn_hidden_size}  ->  {'OK' if self.hidden_matches else 'MISMATCH'}",
            f"  record max token id: {self.record_max_token_id}",
            f"  NextN vocab_size: {self.nextn_vocab_size}  ->  "
            f"{'OK' if self.tokens_within_vocab else 'MISMATCH (token id >= vocab)'}",
        ]
        if self.record_metadata_vocab_size is not None:
            lines.append(
                f"  record metadata vocab_size: {self.record_metadata_vocab_size}  ->  "
                f"{'OK' if self.metadata_vocab_matches else 'MISMATCH vs NextN'}"
            )
        lines.append(f"Overall: {'ALIGNED' if self.ok else 'NOT ALIGNED'}")
        return lines


def load_trace_single_torch_load(path: str | Path, batch_index: int = 0):
    """Load trace / MTP reference with a single ``torch.load`` (for metadata + bundle)."""
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Record is not a dict: {type(payload)}")
    if _is_mtp_reference_payload(payload):
        trace = load_mtp_reference_bundle(payload=payload, batch_index=batch_index)
        meta = payload.get("metadata") or {}
        meta_vocab = meta.get("vocab_size")
        if meta_vocab is None:
            meta_vocab_i = None
        else:
            meta_vocab_i = int(meta_vocab.item()) if hasattr(meta_vocab, "item") else int(meta_vocab)
    else:
        trace = trace_bundle_from_collect_payload(payload)
        meta_vocab_i = None
    return trace, meta_vocab_i


def check_record_nextn_alignment(
    record_path: str | Path,
    *,
    batch_index: int = 0,
    nextn_model_id: str = "lmsys/DeepSeek-R1-NextN",
    trust_remote_code: bool = True,
) -> RecordNextNAlignmentReport:
    path = Path(record_path)
    trace, meta_vocab = load_trace_single_torch_load(path, batch_index=batch_index)

    record_h = int(trace.step_last_hidden.shape[-1])
    all_ids = list(trace.prompt_token_ids) + list(trace.step_next_tokens)
    max_tid = max(all_ids) if all_ids else 0

    cfg = AutoConfig.from_pretrained(nextn_model_id, trust_remote_code=trust_remote_code)
    nextn_h = int(getattr(cfg, "hidden_size", 0))
    nextn_v = int(getattr(cfg, "vocab_size", 0))

    hidden_ok = record_h == nextn_h
    vocab_ok = max_tid < nextn_v
    meta_ok: bool | None = None
    if meta_vocab is not None:
        meta_ok = int(meta_vocab) == nextn_v

    return RecordNextNAlignmentReport(
        record_path=str(path),
        nextn_model_id=nextn_model_id,
        record_hidden_size=record_h,
        record_max_token_id=max_tid,
        record_metadata_vocab_size=int(meta_vocab) if meta_vocab is not None else None,
        nextn_hidden_size=nextn_h,
        nextn_vocab_size=nextn_v,
        hidden_matches=hidden_ok,
        tokens_within_vocab=vocab_ok,
        metadata_vocab_matches=meta_ok,
    )


def assert_record_nextn_aligned(
    record_path: str | Path,
    *,
    batch_index: int = 0,
    nextn_model_id: str = "lmsys/DeepSeek-R1-NextN",
    trust_remote_code: bool = True,
) -> None:
    rep = check_record_nextn_alignment(
        record_path,
        batch_index=batch_index,
        nextn_model_id=nextn_model_id,
        trust_remote_code=trust_remote_code,
    )
    if not rep.ok:
        raise AssertionError("\n".join(rep.summary_lines()))
