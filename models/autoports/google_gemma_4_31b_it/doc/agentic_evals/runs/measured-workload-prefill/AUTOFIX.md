# Diagnostic timeout repair

Implemented one line in tests/check_serving_load_costs.py: urllib urlopen now receives args.request_timeout instead of min(60,args.request_timeout). Existing total elapsed-time validation is unchanged. No numerical model, server, scored harness, task execution or release-limit change.

Dedicated branch fix/diagnostic-request-timeout in /home/mvasiljevic/gemma4-31b-pipeline-v3/metal-diagnostic-timeout; local commit536febd8. Verified commit contains exactly one changed file, one insertion and one deletion. No push. The incomplete no-checkout worktree was not committed: explicit file-path commit bounded the change, and its index was subsequently restored with read-tree HEAD.

check_timeout_regression.py extracts and executes the actual nested request function with mocked HTTP transport. It confirms configured600second and30second values reach urlopen and that an elapsed total deadline still fails. All3checks pass. Repository commit hooks passed. Coordinator is applying the identical one-line fix to the execution checkout for its remaining live slot; this effort does not launch hardware.

Preserve failed measurement as result-first.json. Live long-prefill confirmation remains coordinator-owned; passing host checks proves parameter propagation and elapsed-check retention, not successful device execution or performance. Request timeout is per blocking urllib operation; existing outer timeout remains the hard job bound.
