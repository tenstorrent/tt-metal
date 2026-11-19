# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow and host-boundary tests; never imports TTNN or model code.

AST-load the real generate/output-processing methods into fake device owners.
These checks prove Python stopping/validation and writable host-output semantics,
not device arithmetic, sampler distributions, tokenizer policy or vLLM behavior.
Unknown readiness extras are ignored per Metal Generator ABC; recognized semantic
features must reject when nonneutral. Low-level serving APIs remain strict.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_generate(events):
    path = ROOT / "tt/generator.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    namespace = {
        "torch": torch,
        "ttnn": SimpleNamespace(
            reshape=lambda value, shape: value.reshape(shape),
            pad=lambda value, padding, fill: value,
            copy=lambda source, target: events.append(("copy_logits",)),
            deallocate=lambda value: events.append(("free_logits",)),
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate"]


class FakeGenerator:
    def __init__(self, predicted, *, eos=7, context=64):
        self.events = []
        self.generate = load_generate(self.events).__get__(self)
        self.model = SimpleNamespace(max_seq_len=context, config=SimpleNamespace(eos_token_id=eos))
        self.predicted = iter(predicted)
        self.sampler_logits, self.tokens, self.positions = "logits", "tokens", ["position"]
        self.counters = {"position_refreshes": 0, "token_refreshes": 0}

    def configure_sampling(self, **params):
        self.events.append(("configure", params))

    def prepare(self, **kwargs):
        self.events.append(("prepare", kwargs))

    def reset(self):
        self.events.append(("reset",))

    def prefill_forward(self, tokens, **kwargs):
        assert kwargs == {"prompt_lens": [3], "device_output": True}
        self.events.append(("prefill", tokens.tolist()))
        return [torch.zeros((1, 1, 1, 16))]

    def _copy_host(self, value, target):
        self.events.append(("upload", target, value.tolist()))

    def _sample(self):
        self.events.append(("sample",))

    def _read_tokens(self, rows):
        assert rows == [0]
        self.events.append(("read",))
        return [next(self.predicted)]

    def decode_forward(self, tokens, positions, **kwargs):
        assert tokens is None and positions == [0]
        assert kwargs == {"refresh_inputs": False, "read_tokens": False}
        self.events.append(("decode",))


def stopping_checks():
    cases = [
        ("fixed_length_default", [2, 7, 4, 8], {}, 7, [2, 7, 4, 8]),
        ("scalar_eos", [2, 7, 4, 8], {"stop_on_eos": True}, 7, [2, 7]),
        ("first_token_eos", [7, 2, 4, 8], {"stop_on_eos": True}, 7, [7]),
        ("list_eos", [2, 8, 4, 7], {"stop_on_eos": True}, [7, 8], [2, 8]),
        ("missing_eos", [2, 7, 4, 8], {"stop_on_eos": True}, None, [2, 7, 4, 8]),
        ("explicit_stop_token", [2, 7, 4, 8], {"stop_token_ids": [4]}, 7, [2, 7, 4]),
        ("eos_or_explicit_stop", [2, 4, 7, 8], {"stop_token_ids": [4], "stop_on_eos": True}, 7, [2, 4]),
    ]
    records = []
    for name, predicted, options, eos, expected in cases:
        owner = FakeGenerator(predicted, eos=eos)
        callbacks = []

        def next_input(step, token):
            callbacks.append((step, token))
            return token + 100

        actual = owner.generate([1, 2, 3], 4, next_input=next_input, **options)
        assert actual == expected, (name, actual)
        assert callbacks == list(enumerate(expected)), (name, callbacks)
        assert sum(e[0] == "decode" for e in owner.events) == len(expected) - 1
        assert owner.counters["token_refreshes"] == len(expected) - 1
        assert sum(e[0] == "sample" for e in owner.events) == 1
        records.append({"name": name, "tokens": actual, "decode_calls": len(expected) - 1})
    owner = FakeGenerator([])
    assert owner.generate([1, 2, 3], 0) == [] and owner.events == []
    for options, context in (({"enable_trace": False}, 64), ({}, 4)):
        owner = FakeGenerator([2, 3, 4, 5], context=context)
        try:
            owner.generate([1, 2, 3], 4, **options)
        except ValueError:
            assert owner.events == [], owner.events
        else:
            raise AssertionError("Trace/context rejection did not run before mutation")
    records.append({"name": "zero_count_and_trace_context_guards"})
    return records


def semantic_validation_checks():
    nonneutral = {
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.1,
        "min_p": 0.1,
        "stop": ["done"],
        "min_tokens": 2,
        "logprobs": 3,
        "prompt_logprobs": 0,
        "bad_words": ["bad"],
        "logit_bias": {"2": 1.0},
        "allowed_token_ids": [2, 3],
        "custom_logits_processors": ["processor"],
        "structured_outputs": {"choice": ["yes", "no"]},
    }
    records = []
    for name, value in nonneutral.items():
        owner = FakeGenerator([2, 3, 4, 5])
        try:
            owner.generate([1, 2, 3], 4, **{name: value})
        except (TypeError, ValueError):
            assert owner.events == [], (name, owner.events)
        else:
            raise AssertionError(f"Recognized semantic option {name!r} was silently ignored")
        records.append(name)
    owner = FakeGenerator([2, 3, 4, 5])
    try:
        owner.generate([1, 2, 3], 4, logprobs=0)
    except ValueError:
        assert owner.events == []
    else:
        raise AssertionError("logprobs=0 still requests sampled-token logprobs")
    records.append("logprobs=0")
    # The abstract readiness interface deliberately permits future/model extras.
    owner = FakeGenerator([2, 3, 4, 5])
    assert owner.generate([1, 2, 3], 4, future_model_extra="ignored") == [2, 3, 4, 5]
    return {"nonneutral_rejected_before_mutation": records, "unknown_readiness_extra": "ignored_per_ABC"}


def host_boundary_checks():
    events = []
    path = ROOT / "tt/serving_state.py"
    selected = {"DecodeOutput", "DecodeOutputPool"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in selected]
    namespace = {
        "torch": torch,
        "dataclass": dataclass,
        "ttnn": SimpleNamespace(
            event_synchronize=lambda event: events.append(("sync", event)),
            to_torch=lambda value, **kwargs: value,
            ConcatMeshToTensor=lambda mesh, dim: (mesh, dim),
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    Pool, Output = namespace["DecodeOutputPool"], namespace["DecodeOutput"]
    pool = Pool.__new__(Pool)
    pool.mesh, pool.vocab_size = "fake_mesh", 6
    storage = torch.zeros((1, 1, 32, 8), dtype=torch.bfloat16)
    storage[0, 0, 3, :6] = torch.tensor([-2.0, 1.0, 3.0, 0.0, -1.0, 2.0])
    storage[0, 0, 1, :6] = torch.tensor([4.0, -1.0, 2.0, 0.0, 1.0, 3.0])
    storage[..., 6:] = 1000  # Padding must not escape as vocabulary logits.
    initial = storage.clone()
    handle = Output(pool, 0, 1, (3, 1), False, phase="reading", host=storage, events=("read_done",))
    pool.leases = [handle, None]
    logits = pool.process(handle, is_tokens=False)
    assert logits.shape == (2, 1, 6) and logits.dtype == torch.float32
    assert events == [("sync", "read_done")] and pool.leases == [None, None]
    assert handle.phase == "consumed" and handle.host is None
    expected = initial.reshape(32, 8)[[3, 1], :6].float()
    assert torch.equal(logits[:, 0], expected)
    logits[0, 0, 0] = 99
    assert torch.equal(storage, initial), "Caller wrote into snapshot storage"
    logits[:, 0] = expected
    raw = logits[:, 0].clone()
    assert raw.argmax(-1).tolist() == [2, 0]

    # These are caller-owned transformations of the returned public tensor,
    # not additional generator/device algorithms or a reimplementation of vLLM.
    log_probs = torch.log_softmax(raw, dim=-1)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(2))
    assert log_probs.topk(3).indices.tolist() == [[2, 5, 1], [0, 5, 2]]
    chosen = raw.argmax(-1)
    selected = log_probs.gather(-1, chosen[:, None])
    ranks = (log_probs > selected).sum(-1) + 1
    assert ranks.tolist() == [1, 1] and torch.isfinite(log_probs).all()

    counts = torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    penalty = raw - 0.5 * (counts > 0) - 0.25 * counts
    assert torch.equal(penalty[0], torch.tensor([-2.0, 0.25, 2.0, 0.0, -1.0, 2.0]))
    assert torch.equal(penalty[1], torch.tensor([3.0, -1.0, 2.0, -0.75, 1.0, 3.0]))
    seen = counts > 0
    repeated = torch.where(seen, torch.where(raw >= 0, raw / 2, raw * 2), raw)
    assert torch.equal(repeated[0], torch.tensor([-2.0, 0.5, 1.5, 0.0, -1.0, 2.0]))
    assert torch.equal(repeated[1], torch.tensor([2.0, -1.0, 2.0, 0.0, 1.0, 3.0]))

    features = ["logprobs", "presence_penalty", "frequency_penalty", "repetition_penalty"]
    for name in ("bad_words", "min_tokens", "custom_logits_processors"):
        transformed = raw.clone()
        transformed[:, 2] = -torch.inf
        assert transformed.argmax(-1).tolist() == [5, 0]
        features.append(name)
    biased = raw.clone()
    biased[:, 0] += 10
    assert biased.argmax(-1).tolist() == [0, 0]
    features.append("logit_bias")
    for name, allowed, expected_ids in (
        ("allowed_token_ids", [1, 4], [1, 4]),
        ("structured_outputs", [0, 4], [4, 0]),
    ):
        constrained = raw.clone()
        forbidden = [i for i in range(6) if i not in allowed]
        constrained[:, forbidden] = -torch.inf
        assert constrained.argmax(-1).tolist() == expected_ids
        features.append(name)
    probability = raw.softmax(-1)
    kept = probability >= probability.max(-1, keepdim=True).values * 0.4
    assert kept.tolist() == [[False, False, True, False, False, False], [True, False, False, False, False, False]]
    features.append("min_p")
    assert torch.equal(logits[:, 0], raw) and torch.equal(storage, initial)

    # Source declarations force these feature requests to the full-logit route.
    cap_tree = ast.parse((ROOT / "tt/serving_capabilities.py").read_text())
    cap_class = next(node for node in cap_tree.body if isinstance(node, ast.ClassDef))
    caps = next(
        ast.literal_eval(node.value)
        for node in cap_class.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "serving_capabilities"
    )
    assert caps["supports_device_penalties"] is False and caps["supports_device_grammar"] is False
    return {
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "immutable_snapshot": True,
        "submitted_row_order": [3, 1],
        "caller_transform_features": features,
        "scope": "Host boundary and caller transform examples; not pinned vLLM end-to-end or model accuracy",
    }


def uniform_allocator_checks():
    """Execute the actual uniform wrapper; hardware delegate is separately tested."""
    path = ROOT / "tt/cache_contract.py"
    selected = {"_integer", "_full_layers", "allocate_kv_cache"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in selected]
    calls, marker = [], object()

    def per_layer(model, specs, **kwargs):
        calls.append((model, specs, kwargs))
        return marker

    namespace = {"Integral": Integral, "allocate_kv_cache_per_layer": per_layer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    model = SimpleNamespace(
        layer_indices=[0, 3, 4, 7],
        layers=[
            SimpleNamespace(kind=kind)
            for kind in ("linear_attention", "full_attention", "linear_attention", "full_attention")
        ],
    )
    shape = (32, 1, 32, 256)
    allocate = namespace["allocate_kv_cache"]
    assert allocate(model, shape, "bfloat16", 2, batch=3, context_length=1024) is marker
    assert calls == [(model, [(shape, "bfloat16", 0), (shape, "bfloat16", 1)], {"batch": 3, "context_length": 1024})]
    calls.clear()
    for invalid in (4, 1, -1, True, 2.0):
        try:
            allocate(model, shape, "bfloat16", invalid, batch=3)
        except ValueError:
            assert not calls
        else:
            raise AssertionError(f"Invalid full-attention layer count accepted: {invalid!r}")

    # The reduced device fixture has one full-attention layer. Exercise the
    # actual allocator's cross-layer shared-index branch with fake leaf storage.
    selected = {"_integer", "_context", "allocate_kv_cache_per_layer"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in selected]
    namespace = {"Integral": Integral, "MAX_SLOTS": 32, "normalize_specs": lambda model, specs, **kwargs: specs}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    class Layer:
        def __init__(self, kind):
            self.kind = kind

        def new_recurrent_state(self):
            return {"recurrent": object(), "conv": object()}

        def new_kv_state(self, *, context_length, num_physical_pages, kv_cache=None):
            k, v = (object(), object()) if kv_cache is None else kv_cache
            return {"k": k, "v": v, "page_table": object()}

    shared_model = SimpleNamespace(max_seq_len=1024, layers=[Layer(layer.kind) for layer in model.layers])
    specs = [SimpleNamespace(shape=shape, tensor_idx=17) for _ in range(2)]
    cache = namespace["allocate_kv_cache_per_layer"](shared_model, specs, batch=3)
    assert all(
        cache[layer][slot]["k"] is cache[1][0]["k"] and cache[layer][slot]["v"] is cache[1][0]["v"]
        for layer in (1, 3)
        for slot in range(3)
    )
    assert len({id(cache[layer][slot]["page_table"]) for layer in (1, 3) for slot in range(3)}) == 6
    assert len({id(cache[layer][slot]["recurrent"]) for layer in (0, 2) for slot in range(3)}) == 6
    return {
        "status": "PASS",
        "full_attention_layers": 2,
        "recurrent_layers_excluded": 2,
        "batch_context_and_shape_forwarded_exactly": True,
        "shared_tensor_idx_cross_layer_identity": True,
        "shared_pools_keep_slot_page_and_recurrent_storage_independent": True,
        "scope": "Actual uniform wrapper with fake delegate plus actual per-layer allocation loop with fake leaf storage/normalizer; device allocation/validation separately covered by external_cache_probe",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "status": "RUNNING",
        "scope": "CPU AST fake-owner control flow and actual host-output processing; no TTNN import",
    }
    try:
        result["stopping"] = stopping_checks()
        result["semantic_validation"] = semantic_validation_checks()
        result["host_boundary"] = host_boundary_checks()
        result["uniform_allocator"] = uniform_allocator_checks()
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "tt/generator.py",
                ROOT / "tt/serving_state.py",
                ROOT / "tt/serving_capabilities.py",
                ROOT / "tt/cache_contract.py",
            )
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
