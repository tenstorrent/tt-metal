"""Pin: the emit-e2e scope-grounding gate rejects a hardcoded magic bound that
drives a model iteration (decode horizon, diffusion steps, layer restack, ...)
ONLY when the HF reference config exposes a signal to ground it — and skips
entirely when the reference is unavailable or exposes no such signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.emit_e2e import _scope_grounding_gate

_CFG_STOP = {"gpt_stop_audio_token": 1025, "gpt_max_audio_tokens": 605}
_CFG_DIFFUSION = {"num_train_timesteps": 1000}
_CFG_NO_SIGNAL = {"hidden_size": 512, "vocab_size": 32000}


def _demo(tmp_path: Path, pipeline_src: str) -> Path:
    d = tmp_path / "demo"
    (d / "tt").mkdir(parents=True)
    (d / "tt" / "pipeline.py").write_text(pipeline_src)
    return d


_MAGIC_DECODE = (
    "def run(st):\n"
    "    N = 40\n"
    "    for _ in range(N):\n"
    "        logits = st['fwd'](x)\n"
    "        gen_ids = concat(gen_ids, argmax(logits))\n"
)
_MAGIC_DIFFUSION = (
    "def run(st):\n"
    "    for _ in range(50):\n"
    "        noise = unet(x)\n"
    "        x = scheduler_step(noise, x)\n"
)


def test_fail_magic_decode_when_reference_has_signal(tmp_path: Path) -> None:
    r = _scope_grounding_gate(_demo(tmp_path, _MAGIC_DECODE), reference_config=_CFG_STOP)
    assert r and "scope-grounding" in r


def test_fail_generalized_diffusion_when_reference_has_signal(tmp_path: Path) -> None:
    r = _scope_grounding_gate(_demo(tmp_path, _MAGIC_DIFFUSION), reference_config=_CFG_DIFFUSION)
    assert r and "scope-grounding" in r


def test_skip_when_reference_unavailable(tmp_path: Path) -> None:
    assert _scope_grounding_gate(_demo(tmp_path, _MAGIC_DECODE), reference_config=None) is None


def test_skip_when_reference_has_no_signal(tmp_path: Path) -> None:
    assert _scope_grounding_gate(_demo(tmp_path, _MAGIC_DECODE), reference_config=_CFG_NO_SIGNAL) is None


def test_pass_when_decode_breaks_on_stop(tmp_path: Path) -> None:
    src = (
        "def run(st):\n"
        "    stop = int(model.gpt.stop_audio_token)\n"
        "    for _ in range(4096):\n"
        "        logits = st['fwd'](x); nxt = argmax(logits)\n"
        "        gen_ids = concat(gen_ids, nxt)\n"
        "        if int(nxt) == stop:\n"
        "            break\n"
    )
    assert _scope_grounding_gate(_demo(tmp_path, src), reference_config=_CFG_STOP) is None


def test_pass_when_grounded_via_ar_horizon(tmp_path: Path) -> None:
    src = _MAGIC_DECODE + "    res['ar_horizon'] = first_stop(codes)\n"
    assert _scope_grounding_gate(_demo(tmp_path, src), reference_config=_CFG_STOP) is None


def test_pass_when_bound_from_generation_config(tmp_path: Path) -> None:
    src = (
        "def run(st):\n"
        "    n = model.generation_config.max_new_tokens\n"
        "    for _ in range(n):\n"
        "        logits = st['fwd'](x); gen_ids = concat(gen_ids, argmax(logits))\n"
    )
    assert _scope_grounding_gate(_demo(tmp_path, src), reference_config=_CFG_STOP) is None


def test_pass_when_no_model_iteration_loop(tmp_path: Path) -> None:
    src = "def run(st):\n    for k in range(3):\n        x = plain_add(x)\n    return x\n"
    assert _scope_grounding_gate(_demo(tmp_path, src), reference_config=_CFG_STOP) is None


def test_no_tt_dir_is_safe(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    assert _scope_grounding_gate(tmp_path / "demo", reference_config=_CFG_STOP) is None
