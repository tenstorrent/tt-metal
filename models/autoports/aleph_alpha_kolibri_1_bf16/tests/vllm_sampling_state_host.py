# SPDX-License-Identifier: Apache-2.0
"""Host-only regression for serving seed progression; imports no TTNN or vLLM."""

import ast
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

MODEL = Path(__file__).resolve().parents[1]
PLUGIN = MODEL.parents[2].parent / "vllm-tt-plugin" / "src/vllm_tt_plugin/model_runner.py"


def extract(path, owner, names, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


entropy = []


def random_seed(limit):
    entropy.append(limit)
    return 100 + len(entropy)


env = {"torch": torch, "secrets": SimpleNamespace(randbelow=random_seed)}
extract(MODEL / "tt/generator.py", "KolibriGenerator", {"set_sampling"}, env)
generator_set_sampling = env["set_sampling"]
extract(
    MODEL / "tt/generator_vllm.py",
    "KolibriForCausalLM",
    {"_ensure_sampling_state", "_sampling", "prefill_forward", "decode_forward"},
    env,
)


class FakeGenerator:
    batch_size = 2
    logical_capacity = 64
    traces = {}

    def __init__(self):
        self.counters = Counter()
        self.k = torch.zeros(2, dtype=torch.int32)
        self.p = torch.zeros(2)
        self.temperature = torch.zeros(2)
        self.sampler = SimpleNamespace(_seeds=torch.zeros(2, dtype=torch.int64))
        self.draws = []
        self.active = []

    def copy(self, source, target):
        target.copy_(source)

    set_sampling = generator_set_sampling

    def draw(self, active):
        self.active = active
        self.draws.append([(slot, int(self.sampler._seeds[slot])) for slot in active])
        for slot in active:
            self.sampler._seeds[slot] += 1
        return torch.tensor([7, 8])

    def prefill_forward(self, tokens, *, slots, output_mask, sample_on_device, **kwargs):
        if sample_on_device:
            return self.draw([slot for row, slot in enumerate(slots) if output_mask is None or output_mask[row]])
        return torch.zeros(len(slots), 1, 8)

    def decode_forward(self, tokens, positions, **kwargs):
        active = self.active if positions is None else [i for i, pos in enumerate(positions) if pos >= 0]
        return self.draw(active)


class Adapter:
    host_compat = True
    event_detail = False

    def __init__(self):
        self.generator = FakeGenerator()

    def _prepare(self, cache):
        return self.generator

    def _tables(self, *args):
        return {}

    def event(self, *args, **kwargs):
        pass


for name in ("_ensure_sampling_state", "_sampling", "prefill_forward", "decode_forward"):
    setattr(Adapter, name, env[name])


def params(seeds):
    return SimpleNamespace(top_k=[2] * len(seeds), top_p=[1.0] * len(seeds), temperature=[1.0] * len(seeds), seed=seeds)


def prefill(adapter, end, original, slot, seed, host=False):
    return adapter.prefill_forward(
        torch.arange(end).reshape(1, -1),
        prompt_lens=torch.tensor([end]),
        start_pos=torch.tensor([0]),
        original_prompt_lens=[original],
        empty_slots=[slot],
        prefill_output_mask=[True],
        sampling_params=None if host else params([seed]),
        kv_cache=None,
    )


def decode(adapter, positions, seeds, reload=True, reset=None, remap=None):
    return adapter.decode_forward(
        torch.tensor([[7], [8]]),
        torch.tensor(positions),
        None,
        None,
        sampling_params=params(seeds),
        read_from_device=False,
        reload_inputs=reload,
        reload_sampling_params=reload,
        reset_sampling_state=reload if reset is None else reset,
        slot_remap=remap,
    )


def run():
    adapter = Adapter()
    prefill(adapter, 4, 4, 0, 17)
    decode(adapter, [4, -1], [17, None])
    refreshes = adapter.generator.counters["sampling_parameter_refreshes"]
    decode(adapter, [-999, -999], [17, None], reload=False)
    assert adapter.generator.counters["sampling_parameter_refreshes"] == refreshes
    assert adapter.generator.draws == [[(0, 18)], [(0, 19)], [(0, 20)]]
    prefill(adapter, 7, 7, 1, 29)
    decode(adapter, [7, 6], [29, 17], remap=[1, 0])
    assert adapter.generator.draws[-1] == [(0, 31), (1, 21)]
    decode(adapter, [8, 7], [29, 17], reload=False, reset=True)
    assert adapter.generator.draws[-1] == [(0, 32), (1, 22)]
    print("PASS prefill-first-decode, steady decode, admission, permutation, reset-only")

    prefill(adapter, 10, 4, 0, 17)
    assert adapter.generator.draws[-1] == [(0, 24)]
    decode(adapter, [10, -1], [17, None])
    assert adapter.generator.draws[-1] == [(0, 25)]
    print("PASS preemption replay uses original prompt origin")

    host = Adapter()
    prefill(host, 5, 5, 0, 41, host=True)
    decode(host, [5, -1], [41, None])
    assert host.generator.draws[-1] == [(0, 43)]
    print("PASS host-prefill to device-sampling transition")

    unseeded = Adapter()
    before = len(entropy)
    prefill(unseeded, 4, 4, 0, None)
    initial = unseeded.generator.draws[-1][0][1]
    decode(unseeded, [4, -1], [None, None])
    decode(unseeded, [5, -1], [None, None])
    assert len(entropy) == before + 1
    assert unseeded.generator.draws[-2:] == [[(0, initial + 1)], [(0, initial + 2)]]
    print("PASS unseeded request base assigned once across reloads")

    generator = FakeGenerator()
    generator.set_sampling(seed=[17, 29])
    assert generator.sampler._seeds.tolist() == [18, 30]
    for bad in (-1, 65):
        try:
            generator.set_sampling(seed_offset=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid seed offset accepted")
    print("PASS standalone defaults and offset validation")

    plugin_env = {"torch": torch}
    extract(PLUGIN, "TTModelRunner", {"submit_prefill"}, plugin_env)
    captured = {}
    runner = SimpleNamespace(
        model=SimpleNamespace(
            model_capabilities={"supports_prefill_sampling_origins": True},
            prefill_forward=lambda **kwargs: captured.update(kwargs),
        ),
        kv_caches=None,
        trace_mode="all",
        tt_per_lane_max_num_seqs=2,
        request_specific_rope=False,
        input_batch=SimpleNamespace(req_id_to_index={"a": 0, "b": 1}, num_prompt_tokens=[4, 7]),
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
    )
    model_input = SimpleNamespace(
        input_tokens=None,
        block_tables=None,
        prompt_lens=torch.tensor([10, 12]),
        input_positions=None,
        block_tables_per_layer=None,
        multi_modal_kwargs={},
        row_req_ids=["b", "a"],
        perform_device_sampling=False,
        prefill_empty_slots=None,
    )
    plugin_env["submit_prefill"](runner, model_input, [2])
    assert captured["original_prompt_lens"] == [7, 4]
    runner.model.model_capabilities = {}
    captured.clear()
    plugin_env["submit_prefill"](runner, model_input, [2])
    assert "original_prompt_lens" not in captured
    assert "ttnn" not in sys.modules
    print("PASS plugin capability gate and request-ordered original prompt metadata")
    print("PASS no TTNN imports or hardware calls")


if __name__ == "__main__":
    run()
