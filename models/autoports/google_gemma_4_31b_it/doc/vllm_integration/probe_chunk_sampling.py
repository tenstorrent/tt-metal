# SPDX-License-Identifier: Apache-2.0
"""Host-only reproduction using the pinned plugin's real input builder."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

root = Path(os.environ["VLLM_TT_PLUGIN_ROOT"])
spec = importlib.util.spec_from_file_location("runner_tests", root / "tests/test_model_runner.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
results = []
for mixed in (False, True):
    batch, req = tests._batch_with_one_request(8, 0, 0)
    runner = tests._fake_runner(batch, req)
    runner.check_perform_device_sampling = lambda **kwargs: True
    rows = [("r", 2, 0, 0)]
    if mixed:
        other = tests.CachedRequestState(
            req_id="final",
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=tests.SamplingParams(temperature=0.0),
            generator=None,
            block_ids=([1],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        batch.add_request(other)
        runner.requests["final"] = other
        rows.append(("final", 2, 0, 0))
    model_input = tests._prepare(runner, *rows)
    captured = {}
    runner.model = SimpleNamespace(prefill_forward=lambda **kw: captured.update(kw))
    runner.kv_caches = object()
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.async_decode = SimpleNamespace(note_prefill_submitted=lambda: None)
    tests.TTModelRunner.submit_prefill(runner, model_input, [len(rows)])
    result = dict(
        mixed=mixed,
        starts=model_input.input_positions.tolist(),
        ends=model_input.prompt_lens.tolist(),
        intermediate_mask=model_input.intermediate_prefill_mask.tolist(),
        device_eligible_before_chunk_rule=True,
        perform_device_sampling=model_input.perform_device_sampling,
        adapter_receives_sampling_params="sampling_params" in captured,
        adapter_receives_intermediate_mask="intermediate_prefill_mask" in captured,
    )
    assert result["perform_device_sampling"] is False
    assert not result["adapter_receives_sampling_params"]
    results.append(result)
print(
    json.dumps(
        {
            "scope": "host-only real plugin input preparation and submission; no model/device execution",
            "results": results,
        },
        indent=2,
    )
)
