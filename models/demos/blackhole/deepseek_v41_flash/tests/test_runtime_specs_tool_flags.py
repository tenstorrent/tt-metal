# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU guard of tool calling / reasoning / thinking defaults. No device.

1. Every assets/runtime_spec_*.json (B=4 and B=16) carries the DeepSeek-V4.1 renderer flags and does NOT pass the stock
   vllm_config/chat_template.jinja (which has no tools / tool_calls / tool-result handling).
2. Where the ttis catalog (workflows/model_specs/dev/{llm,vlm}.yaml) has an entry of the same shape, the flag values are equal
   (set TTIS_DIR to a tt-inference-server checkout; skipped otherwise).
3. Offline render of the three message shapes with the DeepSeek-V4.1 encoding (vLLM's vllm.tokenizers.deepseek_v41_encoding if
   importable, else <DSV41_CKPT>/encoding/encoding.py; skipped if neither) contains the tool schema, the assistant call and the
   tool-result marker. The stock jinja template (rendered with jinja2 when available) must NOT, which documents what the
   regression looks like.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_runtime_specs_tool_flags.py
"""

import glob
import importlib.util
import json
import os

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PKG, "assets")
STOCK = os.path.join(PKG, "vllm_config", "chat_template.jinja")
SPECS = sorted(glob.glob(os.path.join(ASSETS, "runtime_spec_*.json")))
REQUIRED = {
    "tokenizer-mode": "deepseek_v41",
    "enable-auto-tool-choice": True,
    "tool-call-parser": "deepseek_v41",
    "reasoning-parser": "deepseek_v41",
    "default-chat-template-kwargs": {"thinking": False},
}


def _norm(k):
    return k.replace("_", "-")


def _vllm_args(path):
    with open(path) as f:
        va = json.load(f)["runtime_model_spec"]["device_model_spec"]["vllm_args"]
    return {_norm(k): v for k, v in va.items()}


def _val(v):
    if isinstance(v, str) and v[:1] in "{[":
        return json.loads(v)
    return v


def test_specs_found():
    # every spec in assets/ is checked below (parametrized over the glob), so a new batch class (B=8) is covered automatically
    assert SPECS, "no assets/runtime_spec_*.json found"
    assert any("_b4_" in s for s in SPECS)
    # set REQUIRE_BATCHES=4,8 (comma list) for a release that ships those batch classes: each must have at least one spec
    for b in [x for x in os.environ.get("REQUIRE_BATCHES", "").split(",") if x]:
        assert any("_b%s_" % b in os.path.basename(s) for s in SPECS), "no runtime spec for B=%s in assets/" % b


@pytest.mark.parametrize("path", SPECS, ids=[os.path.basename(s) for s in SPECS])
def test_spec_has_renderer_flags(path):
    va = _vllm_args(path)
    assert (
        "chat-template" not in va
    ), "stock chat_template.jinja must not be passed (no tools / tool_calls / tool results)"
    for k, want in REQUIRED.items():
        assert k in va, "%s missing from %s" % (k, os.path.basename(path))
        assert _val(va[k]) == want, (k, va[k])


def _catalog_entries(ttis):
    import yaml

    out = []
    for f in ("llm.yaml", "vlm.yaml"):
        with open(os.path.join(ttis, "workflows", "model_specs", "dev", f)) as fh:
            for e in yaml.safe_load(fh)["templates"]:
                if str(e.get("impl", "")).startswith("deepseek_v41"):
                    for d in e["device_model_specs"]:
                        out.append((e["impl"], d))
    return out


def test_every_catalog_entry_has_renderer_flags():
    """The catalog (no runtime spec mounted) is the path we recommend to customers: EVERY DeepSeek-V4.1 entry (B=4, B=8, B=16, ...) must carry the flags and no chat_template."""
    ttis = os.environ.get("TTIS_DIR")
    if not ttis or not os.path.isdir(ttis):
        pytest.skip("set TTIS_DIR to a tt-inference-server checkout")
    pytest.importorskip("yaml")
    entries = _catalog_entries(ttis)
    assert entries, "no deepseek_v41 entries in the catalog"
    for impl, d in entries:
        cv = {_norm(k): v for k, v in d["vllm_args"].items()}
        assert "chat-template" not in cv, (impl, d.get("max_concurrency"), d.get("max_context"))
        for k, want in REQUIRED.items():
            assert k in cv and _val(cv[k]) == want, (impl, d.get("max_concurrency"), d.get("max_context"), k)
    for b in [x for x in os.environ.get("REQUIRE_BATCHES", "").split(",") if x]:
        assert any(str(d["max_concurrency"]) == b for _, d in entries), "no catalog entry for B=%s" % b


@pytest.mark.parametrize("path", SPECS, ids=[os.path.basename(s) for s in SPECS])
def test_spec_flags_equal_catalog(path):
    ttis = os.environ.get("TTIS_DIR")
    if not ttis or not os.path.isdir(ttis):
        pytest.skip("set TTIS_DIR to a tt-inference-server checkout to compare with the catalog")
    pytest.importorskip("yaml")
    va = _vllm_args(path)
    ctx = str(va["max-model-len"])
    matches = [
        (i, d)
        for i, d in _catalog_entries(ttis)
        if str(d["max_context"]) == ctx and str(d["max_concurrency"]) == str(va["max-num-seqs"])
    ]
    if not matches:
        pytest.skip("no catalog entry with this batch size / context")
    for impl, d in matches:
        cv = {_norm(k): v for k, v in d["vllm_args"].items()}
        for k in REQUIRED:
            assert _val(cv[k]) == _val(va[k]), (impl, k, cv[k], va[k])


USER = {"role": "user", "content": "What is the weather in Paris?"}
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
MARK = "ZX-4417-QK"
ASSISTANT = {"role": "assistant", "content": None, "tool_calls": [CALL]}
TOOLMSG = {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temp_c": 17, "station_code": MARK})}


def _encoder():
    try:
        from vllm.tokenizers import deepseek_v41_encoding as enc

        return enc.encode_messages
    except Exception:
        pass
    ckpt = os.environ.get("DSV41_CKPT")
    p = os.path.join(ckpt, "encoding", "encoding.py") if ckpt else None
    if p and os.path.exists(p):
        spec = importlib.util.spec_from_file_location("dsv41_ckpt_encoding", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.encode_messages
    pytest.skip("neither vllm's deepseek_v41_encoding nor $DSV41_CKPT/encoding/encoding.py available")


def test_renderer_encodes_tools_calls_and_results():
    enc = _encoder()
    # tools go on the system message in this encoding
    sys_msg = {"role": "system", "content": "", "tools": TOOLS}
    p1 = enc([sys_msg, USER], thinking_mode="chat")
    assert "get_weather" in p1 and "Get the current weather for a city." in p1, p1[:400]
    p2 = enc([USER, ASSISTANT], thinking_mode="chat")
    assert "get_weather" in p2 and "Paris" in p2, p2[-400:]
    p3 = enc([USER, ASSISTANT, TOOLMSG], thinking_mode="chat")
    assert MARK in p3 and "Paris" in p3, p3[-400:]


def test_stock_template_drops_tools_calls_and_results():
    jinja2 = pytest.importorskip("jinja2")
    with open(STOCK) as f:
        tpl = jinja2.Environment().from_string(f.read())
    try:
        p = tpl.render(
            messages=[USER, dict(ASSISTANT, content=[]), TOOLMSG],
            tools=TOOLS,
            bos_token="<bos>",
            eos_token="<eos>",
            add_generation_prompt=True,
        )
    except Exception as e:  # template needs helpers this minimal env does not have
        pytest.skip("stock template not renderable here: %r" % e)
    assert MARK not in p and "get_weather" not in p, "the stock template unexpectedly encodes tools; update the guard"
