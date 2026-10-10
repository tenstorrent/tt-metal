# Prefix checkpoint CPU foundation

This receipt records the completed local `unittest` command, not a hardware
run. All 21 tests passed. Source hashes are in the experiment evidence index.
Tests use small opaque payloads, a fake device and a real temporary local-file
store; those temporary files are deleted by the test cleanup.

Covered: all-rank byte preservation, logical-to-physical destination page
remapping, divergent branch isolation, matching consumed-token frontiers,
missing logits at an exact-prefix hit, stale namespace/config/layout rejection,
corruption and truncation, cancelled/partial transfer abort, source lease
failure, immutable publication, bounded shared quota, concurrent writers,
eviction waiting on a read lease, and reading from a new Python process.

Not covered: TT DMA implementation, actual GDN numerical continuation, trace
lifetime, vLLM hybrid-state allocation/scheduling, real model restore, storage
bandwidth, cache hit rate, or HTTP TTFT. Serving capability remains disabled.

Reproduce from this worktree without loading the root Metal conftest:

```sh
python3 -m unittest discover \
  -s models/demos/qwen38_27b_qb2/tests/unit \
  -p test_prefix_checkpoint.py -v
```
