# SPDX-License-Identifier: Apache-2.0
"""The final report's batch comes from emit-e2e, persisted under the model's OWN stage file. Reading
it back without naming the model is refused by design, so the report must name it -- otherwise a run
that RECORDED batch 32 prints "batch: not reported; ceilings assume 1" (qwen-image-edit, 2026-10-01).
"""
import cc_optimize.perf_mcp as P
import cc_optimize.summary as S


def test_the_report_names_the_model_when_reading_the_recorded_batch(monkeypatch):
    seen = {}

    def fake_read_stage_batch(model="", task=""):
        seen["model"] = model
        # Mirrors _read_stage_doc: a no-name read is refused (0); a named one finds the recorded batch.
        return 32 if model else 0

    monkeypatch.setattr(P, "read_stage_batch", fake_read_stage_batch)
    monkeypatch.setattr(P, "_model_key", lambda: "qwen_image_edit")
    # Make sure the env fallback can't mask the bug.
    for v in ("TT_PERF_BATCH", "PERF_MCP_BATCH", "TT_PERF_BATCH_SIZE"):
        monkeypatch.delenv(v, raising=False)

    assert S._request_batch() == 32, "a recorded batch must be read back, not reported as 1"
    assert seen["model"] == "qwen_image_edit", "the model must be named when reading the batch"
