# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The resident class's capabilities against each vllm-tt-plugin it may run on (CPU, no device).

Each plugin's own platform._apply_chunked_prefill_policy resolves the scheduler config from the capabilities the class
declares under that plugin (resident_capabilities, evaluated in a process that imports that plugin, as vLLM does):
- max_num_seqs 1, any plugin: chunked prefill off and the token budget raised to max_model_len (one user's prompt runs
  whole, as before the multi-user class existed) - review finding 1: under upstream da7c018 the chunk keys turned
  chunking on with an 8192-token budget and the single-user prefill raised on the first continuation;
- several users: aligned, exclusive chunks of the PP unit where the plugin has the TT chunk core (resident-mu), whole
  prompts where it does not (da7c018).

usage: PYTHONPATH=<tt-metal> python -m pytest --noconftest -c /dev/null qwen36_resident/tests/test_vllm_rows_plugins.py
env: RESIDENT_PLUGIN_UPSTREAM_SRC, RESIDENT_PLUGIN_MU_SRC (plugin src dirs), RESIDENT_VLLM_PYTHON (a python with vLLM)
"""
import json
import os
import subprocess

import pytest

R = "/home/ttuser/experiments/qwen36_27b"
PLUGINS = {
    "upstream-da7c018": os.environ.get("RESIDENT_PLUGIN_UPSTREAM_SRC", f"{R}/vllm-tt-plugin-resident/src"),
    "resident-mu": os.environ.get("RESIDENT_PLUGIN_MU_SRC", f"{R}/vllm-tt-plugin-resmu/src"),
}
PYTHON = os.environ.get("RESIDENT_VLLM_PYTHON", f"{R}/.venv_vllm_server/bin/python")
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_MODEL_LEN = 32768

PROBE = (
    """
import json, sys
from types import SimpleNamespace as NS
import vllm  # noqa: F401  (vLLM first: its bootstrap imports the platform plugin)
import vllm_tt_plugin, vllm_tt_plugin.platform as P
from qwen36_resident.vllm_rows import resident_capabilities

class Qwen36ResidentForCausalLM:
    model_capabilities = resident_capabilities()

out = dict(plugin=vllm_tt_plugin.__file__, caps=Qwen36ResidentForCausalLM.model_capabilities, seqs={})
for seqs in (1, 8):
    sc = NS(enable_chunked_prefill=True, max_num_batched_tokens=8192, long_prefill_token_threshold=0, max_num_seqs=seqs,
            disable_chunked_mm_input=False, max_num_scheduled_tokens=None, async_scheduling=True)
    cfg = NS(scheduler_config=sc, model_config=NS(max_model_len=%d), additional_config={},
             parallel_config=NS(data_parallel_size=1), kv_transfer_config=None)
    P._apply_chunked_prefill_policy(cfg, Qwen36ResidentForCausalLM.model_capabilities, Qwen36ResidentForCausalLM)
    policy = getattr(__import__("vllm_tt_plugin.config", fromlist=["x"]), "get_tt_prefill_chunk_policy", None)
    pol = policy(cfg) if policy else None
    out["seqs"][seqs] = dict(chunked=sc.enable_chunked_prefill, budget=sc.max_num_batched_tokens,
                             threshold=sc.long_prefill_token_threshold,
                             policy=None if pol is None else [pol.chunk_tokens, pol.exclusive])
print("RESULT " + json.dumps(out))
"""
    % MAX_MODEL_LEN
)


def resolve(plugin_src):
    if not os.path.isdir(plugin_src) or not os.path.exists(PYTHON):
        pytest.skip(f"no plugin at {plugin_src} or no vLLM python at {PYTHON}")
    env = dict(os.environ, PYTHONPATH=f"{plugin_src}:{WT}", HF_HUB_OFFLINE="1", VLLM_PLUGINS="tt,tt_model_registry")
    p = subprocess.run([PYTHON, "-c", PROBE], env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"probe failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}"
    out = json.loads(line[len("RESULT ") :])
    assert out["plugin"].startswith(plugin_src), f"imported {out['plugin']}, not {plugin_src}"
    return out


@pytest.mark.parametrize("plugin", PLUGINS)
def test_one_user_resolves_as_before_on_every_plugin(plugin):
    one = resolve(PLUGINS[plugin])["seqs"]["1"]
    assert one["chunked"] is False and one["budget"] == MAX_MODEL_LEN and one["threshold"] == 0, one


def test_several_users_chunk_only_with_the_chunk_core():
    up, mu = resolve(PLUGINS["upstream-da7c018"]), resolve(PLUGINS["resident-mu"])
    assert "tt_prefill_chunk_tokens" not in up["caps"] and "supports_chunked_prefill" not in up["caps"]
    assert up["seqs"]["8"]["chunked"] is False and up["seqs"]["8"]["budget"] == MAX_MODEL_LEN
    assert mu["caps"]["tt_prefill_chunk_tokens"] == 1024 and mu["caps"]["tt_prefill_exclusive_chunks"] is True
    eight = mu["seqs"]["8"]
    assert eight["chunked"] is True and eight["policy"] == [1024, True], eight
    assert eight["budget"] >= MAX_MODEL_LEN + 8 * 1024, eight


def test_the_old_declaration_breaks_one_user_on_upstream():
    """negative control (the reviewer's probe_old_plugin.py): the static chunk keys under da7c018 at 1 user"""
    src = PLUGINS["upstream-da7c018"]
    if not os.path.isdir(src):
        pytest.skip("no upstream plugin")
    probe = PROBE.replace(
        "model_capabilities = resident_capabilities()", "model_capabilities = resident_capabilities(chunk_core=True)"
    )
    env = dict(os.environ, PYTHONPATH=f"{src}:{WT}", HF_HUB_OFFLINE="1", VLLM_PLUGINS="tt,tt_model_registry")
    p = subprocess.run([PYTHON, "-c", probe], env=env, capture_output=True, text=True, timeout=600)
    line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
    one = json.loads(line[len("RESULT ") :])["seqs"]["1"]
    assert one["chunked"] is True and one["budget"] == 8192, one
