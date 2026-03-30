# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Default filesystem paths for record-based CPU demos in this workspace.

NextN draft weights always come from Hugging Face Hub (``lmsys/DeepSeek-R1-NextN``) via
``snapshot_download`` in the run scripts — not from a local path here.

**Default record path** (MTP reference ``.pt``):

  ``/proj_sw/user_dev/dchrysostomou/deepseek-v3-cache/test_io_cache/mtp_full_model_seq128.pt``

**Minimal bundle** (this folder’s ``run_mtp_from_record_cpu.py``; defaults match below)::

  cd speculative_from_record_deepseek_r1
  python run_mtp_from_record_cpu.py

**Full tt-metal tree** (same logic; more scripts)::

  export PYTHONPATH=/proj_sw/user_dev/dchrysostomou/tt-metal:$PYTHONPATH
  python models/demos/speculative_deepseek_r1_broad/scripts/run_nextn_mtp_from_record_cpu.py

Add ``--quiet`` to the Python script to trim startup lines and Hub progress bars.

**Explicit record path**::

  python run_mtp_from_record_cpu.py \\
    --record /proj_sw/user_dev/dchrysostomou/deepseek-v3-cache/test_io_cache/mtp_full_model_seq128.pt

``--embed-head-aux-safetensors`` is **only** needed if ``nextn_layer_parameters.safetensors``
does not already contain both token embeddings and ``shared_head.head``.
Build that file once with ``scripts/materialize_nextn_embed_head_aux_from_r1_shards.py`` in this folder
(two Hub shards from ``deepseek-ai/DeepSeek-R1-0528``). Default output path: ``DEFAULT_EMBED_HEAD_AUX_PATH``.

**Disk:** ``DEFAULT_HF_HOME`` is on WEKA (``hf_cache`` on proj_sw). NextN scripts always call
``set_hf_home`` for that path unless ``--hf-home`` is passed (so a shell ``HF_HOME`` under
``/home`` cannot redirect Hub downloads there). Records stay under ``deepseek-v3-cache/``.
"""

from __future__ import annotations

from pathlib import Path

# MTP reference .pt (recorded base hidden states + greedy tokens + optional logits)
DEFAULT_MTP_RECORD_PATH = Path(
    "/proj_sw/user_dev/dchrysostomou/deepseek-v3-cache/test_io_cache/mtp_full_model_seq128.pt"
)

NEXTN_HF_REPO_ID = "lmsys/DeepSeek-R1-NextN"

# Weight repos (``lmsys/DeepSeek-R1-NextN``, ``lmsys/DeepSeek-V3-NextN``) ship config + fusion + tokenizer
# but omit ``modeling_deepseek.py``; ``SGLang/DeepSeek-V3-NextN`` publishes the matching remote-code module.
NEXTN_MODELING_AUX_REPO_ID = "SGLang/DeepSeek-V3-NextN"

# Hugging Face cache on WEKA — avoids default ``~/.cache/huggingface`` when ``$HOME`` is a small
# full partition (NextN / tokenizer downloads will fail with “no space left on device”).
DEFAULT_HF_HOME = Path("/proj_sw/user_dev/dchrysostomou/hf_cache")

# Optional output from ``materialize_nextn_embed_head_aux_from_r1_shards.py`` (two HF shards → aux file).
DEFAULT_EMBED_HEAD_AUX_PATH = DEFAULT_HF_HOME / "embed_head_aux_deepseek_r1_0528.safetensors"
