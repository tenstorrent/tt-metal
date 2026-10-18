"""Execute actual scheduler/EOS methods with device calls mocked; no TTNN import."""

import argparse
import ast
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/scheduler_cpu.json")
    args = parser.parse_args()
    source = ast.parse((ROOT / "tt/generator.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    namespace = dict(
        Generator=object,
        torch=torch,
        Counter=Counter,
        time=time,
        ttnn=SimpleNamespace(to_torch=lambda x: x, get_device_tensors=lambda x: [x]),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "generator.py", "exec"), namespace)
    kind = namespace[cls.name]
    g = kind.__new__(kind)
    g.batch_size = 3
    g.model = SimpleNamespace(max_requests=5, context=2048, scratch_page_base=192)
    g.counters = Counter()
    g.sampling_mode = "device"
    g.kv_cache = object()
    g.page_table = torch.arange(192, dtype=torch.int32).reshape(3, 64)
    pages = g.page_table.flip(1)
    g._page_host = g.page_table.clone()
    g._request_page_host = pages.clone()
    g._request_slot_host = [4, 3, 2]
    g._active = [True] * 3
    for name in ("tt_slots", "tt_page", "tt_active", "tt_logit_mask", "tt_pos", "tt_token"):
        setattr(g, name, name)
    copied = {}
    g._copy = lambda value, target: copied.__setitem__(target, value.clone())
    g._bind_cache = lambda cache: None
    g.prepare = lambda: None
    g._replay = lambda mode: None
    for activity in ([True, True, False], [True, True, True]):
        g.decode_forward(
            None, [7, 31, 0], page_table=None, kv_cache=g.kv_cache, active_rows=activity, read_tokens=False
        )
        expected = [4, 3, 2 if activity[2] else 7]
        assert copied["tt_slots"][:, 0].tolist() == expected
        assert torch.equal(g._request_page_host, pages)
        assert g._request_slot_host == [4, 3, 2]
    assert torch.equal(copied["tt_page"], pages)
    before = dict(g.counters)
    g.decode_forward(None, None, page_table=None, kv_cache=g.kv_cache, read_tokens=False)
    assert dict(g.counters) == before

    checkpoint = "/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540"
    config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    assert config.eos_token_id == [248046, 248044]
    sequence = [10, 248046, 20, 248044, 30]
    results = {}
    for mode in ("device", "host"):
        for stop in (False, True):
            e = kind.__new__(kind)
            e.model = SimpleNamespace(context=2048, generation_config=config)
            e.batch_size = 1
            e.sampling_mode = mode
            e.counters = Counter()
            e.page_table = None
            e.kv_cache = None
            e.prepare = e.reset = lambda: None
            e._read_tokens = lambda: [10]
            e.tt_history = torch.tensor([[99] + sequence])
            outputs = iter(sequence)

            def logits(*args, **kwargs):
                value = next(outputs)
                scores = torch.zeros(1, 248320)
                scores[0, value] = 1
                return scores

            e.prefill_forward = logits
            e.decode_forward = logits
            result = e.generate([99], 5, stop_on_eos=stop)
            assert result == (sequence[:2] if stop else sequence), (mode, stop, result)
            results[f"{mode}_stop_{stop}"] = result
    report = dict(
        status="pass",
        scope="CPU mocked device calls; actual generator methods",
        retained_scheduler_bindings=True,
        unchanged_decode_has_no_refresh=True,
        canonical_eos=config.eos_token_id,
        generation_outputs=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
