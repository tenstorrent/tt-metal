# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RUN-first regression target for the DiffusionGemma text demo (#47464).

This pins the canonical short-prompt, small-context multi-block hardware run
(R-b) that passed on QB2 2026-07-02 as a reproducible, device-gated smoke test
so future changes cannot silently break "prompt -> committed blocks -> decoded
text" without crashing. Output *correctness* is explicitly deferred (#48291);
the RUN gate is a clean exit + the ``DG_TEXT_DEMO_SUCCESS`` marker, not text
quality. The test disables EOS stopping so degenerate EOS-heavy output still
commits the requested two blocks and exercises cross-block KV / position
advancement. ``max_seq_len`` must cover ``prompt + 2 * canvas``; use 1024 so
the second block's RoPE/cache slices do not overrun after the 32-token-aligned
prompt prefix.

Run on QB2::

    source /home/zni/venvs/tt-diffusion-gemma/bin/activate
    export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn TT_METAL_HOME=/home/zni/tt-metal
    export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal MESH_DEVICE=P150x4 DG_RUN_DEVICE=1
    pytest models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py -q

Override the checkpoint / mesh with ``DG_CKPT`` / ``MESH_DEVICE``; set
``DG_TEXT_DEMO_NUM_LAYERS`` for a cheaper reduced-depth short-prompt smoke.
The long-prompt smoke defaults to one layer; set
``DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS=full`` to run all layers.
The 256K-allocation smoke also defaults to one layer; set
``DG_TEXT_DEMO_256K_NUM_LAYERS=full`` to run all layers. The full-depth
256K variant uses ``--argmax-sampling`` because the default full-vocab device
Gumbel allocation is a known DRAM-fragmentation OOM at that size.
"""

import os

import pytest

from models.experimental.diffusion_gemma.demo import text_demo

pytestmark = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)

_DEFAULT_CKPT = "/home/zni/dg_models/diffusiongemma-26B-A4B-it"


def _checkpoint() -> str:
    return os.environ.get("DG_CKPT", _DEFAULT_CKPT)


def _require_local_checkpoint() -> str:
    checkpoint = _checkpoint()
    if not os.path.isdir(checkpoint):
        pytest.skip(f"checkpoint not found at {checkpoint!r}; set DG_CKPT")
    return checkpoint


def test_short_prompt_two_block_run_exits_clean(monkeypatch):
    """R-b: short prompt -> 2 committed 256-token blocks -> decoded text, no crash."""
    checkpoint = _require_local_checkpoint()
    info_lines: list[str] = []
    monkeypatch.setattr(text_demo.logger, "info", lambda message: info_lines.append(str(message)))
    argv = [
        "--checkpoint",
        checkpoint,
        "--local-files-only",
        "--mesh",
        os.environ.get("MESH_DEVICE", "P150x4"),
        "--max-seq-len",
        "1024",
        "--canvas-length",
        "256",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "512",
        "--num-blocks",
        "2",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_NUM_LAYERS")
    if num_layers is not None:
        argv += ["--num-layers", num_layers]

    assert text_demo.main(argv) == 0
    success_lines = [line for line in info_lines if line.startswith("DG_TEXT_DEMO_SUCCESS ")]
    assert len(success_lines) == 1
    summary = text_demo._parse_success_summary(success_lines[0])
    assert summary["generated_tokens"] == 512
    assert summary["blocks"] == 2
    assert summary["prompt_len"] == 32
    assert summary["next_pos"] == 544


def test_short_prompt_256k_context_allocation_exits_clean(monkeypatch):
    """Short prompt with a 256K context allocation, without a huge input prompt."""
    checkpoint = _require_local_checkpoint()
    info_lines: list[str] = []
    monkeypatch.setattr(text_demo.logger, "info", lambda message: info_lines.append(str(message)))
    argv = [
        "--checkpoint",
        checkpoint,
        "--local-files-only",
        "--mesh",
        os.environ.get("MESH_DEVICE", "P150x4"),
        "--max-seq-len",
        "262144",
        "--canvas-length",
        "32",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "32",
        "--num-blocks",
        "1",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_256K_NUM_LAYERS", "1")
    if num_layers.lower() != "full":
        argv += ["--num-layers", num_layers]
    else:
        argv += ["--argmax-sampling"]

    assert text_demo.main(argv) == 0
    success_lines = [line for line in info_lines if line.startswith("DG_TEXT_DEMO_SUCCESS ")]
    assert len(success_lines) == 1
    summary = text_demo._parse_success_summary(success_lines[0])
    assert summary["generated_tokens"] == 32
    assert summary["blocks"] == 1
    assert summary["prompt_len"] == 32
    assert summary["next_pos"] == 64


def test_long_prompt_two_block_maskless_run_exits_clean(monkeypatch):
    """R-a: prompt long enough to force maskless denoise -> 2 blocks, no crash."""
    checkpoint = _require_local_checkpoint()
    info_lines: list[str] = []
    monkeypatch.setattr(text_demo.logger, "info", lambda message: info_lines.append(str(message)))
    argv = [
        "--checkpoint",
        checkpoint,
        "--local-files-only",
        "--mesh",
        os.environ.get("MESH_DEVICE", "P150x4"),
        "--prompt",
        "hello " * 1000,
        "--max-seq-len",
        "1536",
        "--canvas-length",
        "32",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "64",
        "--num-blocks",
        "2",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS", "1")
    if num_layers.lower() != "full":
        argv += ["--num-layers", num_layers]

    assert text_demo.main(argv) == 0
    success_lines = [line for line in info_lines if line.startswith("DG_TEXT_DEMO_SUCCESS ")]
    assert len(success_lines) == 1
    summary = text_demo._parse_success_summary(success_lines[0])
    assert summary["generated_tokens"] == 64
    assert summary["blocks"] == 2
    assert summary["prompt_len"] == 1024
    assert summary["next_pos"] == 1088
