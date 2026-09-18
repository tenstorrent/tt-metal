"""Exercise the actual repository pure runtime constructors with lightweight model stand-ins."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "tt"
PACKAGE = "_capacity_bound_runtime"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".tt_prefill_runtime", SOURCE / "tt_prefill_runtime.py")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


def params(capacity):
    return types.SimpleNamespace(
        max_seq_len=capacity,
        chunk_size=1024,
        num_users=2,
        num_layers=32,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        mesh_shape=(4, 8),
        sp_axis=0,
        tp_axis=1,
        use_trace=False,
        dflash_enabled=False,
        tp_shard_kv=False,
    )


class Model:
    def __init__(self, device, checkpoint, **kwargs):
        self.num_layers = kwargs["num_layers"]
        self.max_seq_len = kwargs["max_seq_len"]
        self.kwargs = kwargs
        self.calls = []

    def prefill_chunk(self, tokens, cache, **kwargs):
        self.calls.append(kwargs)
        return None


class RuntimeCapacityTests(unittest.TestCase):
    # The real build_runtime must pass the chosen capacity into both the model and upload helper.
    def test_actual_constructor_propagates_all_requested_capacities(self):
        model = types.ModuleType(PACKAGE + ".model")
        model.PrefillModel = Model
        inputs = types.ModuleType(PACKAGE + ".input")
        inputs.upload_token_chunk = lambda device, ids, **kw: dict(ids=ids, **kw)
        api = types.ModuleType("ttnn")
        api.bfloat8_b = "BFP8"
        api.synchronize_device = lambda device: None
        with patch.dict(sys.modules, {PACKAGE + ".model": model, PACKAGE + ".input": inputs, "ttnn": api}):
            for value in (4096, 8192, 16384, 32768, 65536):
                bound = runtime.build_runtime(object(), params=params(value), checkpoint_path=Path("/unused"))
                self.assertEqual(bound.model.max_seq_len, value)
                self.assertEqual(
                    bound.model.kwargs, dict(num_layers=32, enable_lm_head=False, cache_dtype="BFP8", max_seq_len=value)
                )
                self.assertEqual(bound.geometry.cache_shape, (64, 1, value // 4, 128))
                self.assertEqual(bound.geometry.rope_local_sequence, (value + 1024) // 4)
                packet = bound.make_chunk_input([19] * 992, actual_start=value - 1024)
                self.assertEqual(
                    (packet["actual_start"], packet["actual_end"], packet["max_seq_len"]),
                    (value - 1024, value - 32, value),
                )
                cache = types.SimpleNamespace(num_users=2, num_layers=32, max_seq_len=value, sp=4)
                bound._check_cache(cache)
                cache.max_seq_len = 2048
                with self.assertRaises(ValueError):
                    bound._check_cache(cache)

    # Invalid geometry is refused by the actual config before any model constructor or tensor import.
    def test_invalid_capacity_and_mismatched_model_fail_before_execution(self):
        for value in (True, 4097, 262144):
            with self.assertRaises((ValueError, TypeError)):
                runtime.config_from_params(params(value))
        config = runtime.config_from_params(params(65536))
        with self.assertRaisesRegex(ValueError, "model capacity"):
            runtime.TtPrefillRuntime(
                None,
                config=config,
                model=types.SimpleNamespace(num_layers=32, max_seq_len=2048),
                synchronize=lambda device: None,
                upload=lambda *a, **k: None,
            )

    # Serving starts with request0 after compile-style warmup; callbacks follow synchronization.
    def test_actual_runtime_warmup_keeps_first_serving_id_available(self):
        from capacity_warmup import warmup_geometry
        from checks_capacity_warmup import Chunk, fixture

        doc, tokens = fixture()
        events = []
        config = runtime.config_from_params(params(4096))
        model = Model(None, None, num_layers=32, max_seq_len=4096)
        bound = runtime.TtPrefillRuntime(
            None,
            config=config,
            model=model,
            synchronize=lambda device: events.append("sync"),
            upload=lambda ids, **kw: Chunk(ids),
        )
        cache = types.SimpleNamespace(num_users=2, num_layers=32, max_seq_len=4096, sp=4)
        bound.compile(cache)
        warmup_geometry(bound, cache, doc, tokens, lambda: None, lambda row: None)
        self.assertEqual(bound._last_request_id, -1)
        self.assertEqual(len(model.calls), 10)
        bound.set_layer_completion_sink(lambda layer, request: events.append((layer, request)))
        packet = bound.make_chunk_input(tokens["0"][:1024], actual_start=0)
        bound.prefill_chunk(packet, cache, slot_id=0, actual_start=0, actual_end=1024, request_id=0)
        self.assertEqual(events[-33], "sync")
        self.assertEqual(events[-32:], [(i, 0) for i in range(32)])
        self.assertTrue(all(x["skip_lm_head"] is True for x in model.calls))


if __name__ == "__main__":
    unittest.main()
