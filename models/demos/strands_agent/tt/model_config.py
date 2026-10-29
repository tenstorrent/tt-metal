# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from pathlib import Path

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.strands_agent.tt.weight_mapping import (
    STRANDS_BASE_HF_MODEL,
    load_strands_state_dict,
    resolve_checkpoint,
)


def adapter_revision(adapter_dir) -> str:
    """HF snapshot commit hash when the adapter sits in a hub snapshot, else a hash of the adapter file's size+mtime."""
    adapter_dir = Path(adapter_dir)
    if adapter_dir.parent.name == "snapshots":
        return adapter_dir.name
    st = (adapter_dir / "lora" / "adapter_model.safetensors").stat()
    return hashlib.sha256(f"{adapter_dir.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]


class StrandsDeciderModelArgs(Qwen36ModelArgs):
    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=4096, **kwargs):
        # The text tower is Qwen3.5-2B-Base; the LoRA adapter is merged in load_state_dict. The base has no
        # usable MTP head here (mtp.* is skipped), so MTP must be off.
        os.environ.setdefault("HF_MODEL", STRANDS_BASE_HF_MODEL)
        # Precision defaults for this model (qwen36 keeps both off); an explicit env value overrides them.
        os.environ.setdefault("QWEN36_GDN_FUSED_PREFILL", "1")
        os.environ.setdefault("QWEN36_MLP_GATEUP_DTYPE", "bfp8")
        os.environ.setdefault("QWEN36_PREFILL_MINIMAL_MATMUL", "1")
        os.environ.setdefault("QWEN36_PREFILL_FUSED_SWIGLU", "1")
        super().__init__(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, enable_mtp=False, **kwargs
        )
        self.adapter_dir = resolve_checkpoint()
        # Isolate the TT weight cache: weight_cache_path() is rooted at model_cache_path, which (with or without
        # TT_CACHE_PATH) is shared with plain Qwen3.5-2B-Base. Merged weights must never be loaded from, or
        # written to, that cache, so add a per-adapter-revision subdir.
        self.model_cache_path = self.model_cache_path / "strands_decider" / adapter_revision(self.adapter_dir)

    def load_state_dict(self):
        return load_strands_state_dict(self.CKPT_DIR, self.adapter_dir / "lora")
