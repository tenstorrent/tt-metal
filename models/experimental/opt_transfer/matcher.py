import json
import os
import re
from models.experimental.opt_transfer.schema import FusionProposal
from models.experimental.opt_transfer.config import CONFIG

SYSTEM = (
    "You map a PyTorch op graph to fused TTNN ops by transferring optimizations from a "
    "knowledge base. For each applicable subsequence (chain) or sibling-branch group "
    "(horizontal_merge), emit a FusionProposal that names the matched node(s), the fused op, "
    "a config (use {DIM} placeholders), and the weight_transform if weights must be folded. "
    "Only propose fusions that preserve the model's computation. Return a JSON list of "
    "FusionProposal objects and nothing else."
)


def _extract_json(text: str):
    """Tolerate ```json fences / surrounding prose: parse the first JSON array/object found."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


class _AnthropicTransport:
    def __init__(self, model):
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def create(self, **kwargs):
        msg = self._client.messages.create(model=self._model, max_tokens=4096, **kwargs)
        return {"content": [{"type": "text", "text": b.text} for b in msg.content if b.type == "text"]}


EXTRACT_SYSTEM = (
    "Given a ttnn op, its registered golden source (if any), unit-test examples, and model "
    "call sites, emit KBEntry JSON dicts. The torch_pattern MUST be taken from the golden/test "
    "source (the unoptimized subsequence the op replaces) — do not invent it. Fill pattern_kind, "
    "config_template (use {DIM} placeholders), weight_transform, category. "
    "ALSO emit placement_observations: for each tensor whose "
    "memory_config/program_config and size regime you can read from the call site or test, add "
    "{op, tensor_role, size_descriptor, memory_config:{buffer:'L1'|'DRAM',layout,shard_spec_template}, "
    "program_config, condition:{var,op,value} or null, source}. Capture size-conditional placement "
    "(e.g. 'L1 if seq<=1024 else DRAM') as a condition. Return a JSON list only."
)


class LLMClient:
    """Production client for BOTH the KB miner (extract_entries) and the matcher (propose),
    sharing prompt-cached context. Transport is injectable for offline tests."""

    def __init__(self, transport=None, model=None):
        self.transport = transport or _AnthropicTransport(model or CONFIG.matcher_model)

    def _complete(self, system_blocks, user_text):
        resp = self.transport.create(system=system_blocks, messages=[{"role": "user", "content": user_text}])
        return resp["content"][0]["text"]

    def extract_entries(self, op, available, used, golden_src):
        sys = [{"type": "text", "text": EXTRACT_SYSTEM}]
        user = json.dumps(
            {
                "op": op,
                "golden_src": golden_src,
                "test_examples": available.get("examples", []),
                "call_sites": [u["snippet"] for u in used],
            },
            indent=2,
        )
        return _extract_json(self._complete(sys, user))

    def propose(self, graph_summary, kb, diagnosis=None):
        kb_text = json.dumps([e.to_dict() for e in kb], indent=2)
        sys = [
            {"type": "text", "text": SYSTEM},
            {
                "type": "text",
                "text": "KNOWLEDGE BASE:\n" + kb_text,
                "cache_control": {"type": "ephemeral"},
            },  # reused across blocks + repair iters
        ]
        payload = {"op_graph": graph_summary}
        if diagnosis:
            payload["prior_failure_diagnosis"] = diagnosis  # re-propose using KB applicability_notes
        return [FusionProposal(**d) for d in _extract_json(self._complete(sys, json.dumps(payload, indent=2)))]
