# Benchmark integration repairs

## Event-query binding deployment

Failing contract: a fresh server interpreter must expose the nonblocking `ttnn.event_query` API built from the current source. The first probe failed during import, before device open, despite a successful native build. AutoDebug independently identified the stale selected source-tree extension; see `AUTODEBUG.md` and `autodebug_import.log`.

Priority hypotheses: (1) stale deployed extension, (2) missing native registration, (3) dependency ABI mismatch. Static import resolution and binary binding literals discriminate (1); the source registers the function and the freshly built binary contains it. Applied the existing `cmake --install build_Release --component tt_pybinds` deployment component, rather than changing correct binding source. Fresh interpreter import then passed and resolved `ttnn/ttnn/_ttnn.so`. Original failed log remains `event_query_smoke.log`; install/import evidence is in `event_query_install.log` and `event_query_import.log`.

The added binding calls the existing nonblocking MeshEvent query, releases the GIL and reads completed-event counters over every mesh device. It enqueues no barrier and does not synchronize the serving path. Core source checkpoint: `21fddded`. Build and repository hooks passed. The separate bounded four-chip add/query/readback smoke is retained in `event_query_smoke_retry.log`; its actual result, rather than import success, establishes hardware operation.

## Inherited CLI spelling and configuration recorder

The initial 32-slot server completed startup and a 128/4 completion probe, then `server_control.py` failed while looking for `--max-num-seqs`. Its actual process argv used the equivalent `--max_num_seqs` spelling emitted by the inherited readiness launcher (and `--max_model_len` for context). The direct `/proc/<api_pid>/cmdline` observation in the setup configuration snapshot discriminates this recorder defect from a server/capacity failure. No request or model execution failed.

The recorder now verifies either observed spelling and preserves the original argv. Future launches also explicitly supply the canonical hyphenated flags with identical values, satisfying the final benchmark checker's literal command requirement. The initial server is used only for setup probes; it will be drained/stopped before the timed run, which launches the corrected command. `setup/initial_server.log` retains the failure; `setup/initial_identity_retry.log` and the collector readiness output verify the repair on the existing setup server. Prefix caching, model capacity, precision and scheduling behavior are unchanged.

Exact performance transport: upstream random text roundtrips expanded some 4096-token inputs. Model-local bench_exact.py preserves the original token-ID prompts through the unmodified vLLM 0.26 timing client. CPU tests and a live 32-request preflight passed: every request 4096/128, 131072 input and 4096 output tokens total, 116.736 s; full-phase collector also passed. Original mismatch artifacts remain in setup/perf-probe/. See setup/EXACT_TRANSPORT_AUTOFIX.md.

Template verifier initially assumed unbatched serialized prompts and a list return from Transformers 5.12. It now unwraps the upstream one-request batch and explicitly requests return_dict=False. All 18 saved prompts match the native server template tokens and response prompt usage. Failed verifier logs remain; model requests were unchanged.
