# Artifact-checker package context repair

The first direct invocation of tests/check_optimized_multichip_evidence.py failed before report emission: validation_suite loads run_optimized_multichip_validation.py with runpy.run_path and no qualified run_name. On the resource job, `from .check_optimized_multichip_evidence import validate_fused_decode_norm_resources` raises `ImportError: attempted relative import with no known parent package`.

`check_final29.py` is the CPU-only final entrypoint. It supplies the planner's qualified run_name for that exact path and calls the unchanged checker.main. Every checker function, assertion, source hash, numerical threshold and artifact remains unchanged. The adapter restores runpy.run_path after use and asserts no torch/ttnn import. This avoids changing any frozen measured source or manufacturing new device receipts. A future source cleanup can put this same qualified run_name directly in the checker.

Command: `python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/check_final29.py --manifest models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/final_manifest.json`.

The repaired entrypoint completes every group. Its actual pre-review output in final_gate.json has25 passed groups and two explicit errors: the preserved B1 confidence outcome and pending independent review/commit. It exits1, with no stderr, rather than suppressing either finding. These are not numerical hardware failures; all68 validation jobs and34 performance/profile jobs retain passing receipts. Final review must classify the performance uncertainty against the actual user/skill contract. Local commit follows a passing review, per the stage-review skill.
