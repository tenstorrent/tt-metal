# CI phase accounting confirmation

Goal: verify the normal SWE CI run 42200513517 retained automatic model-owned phase evidence. Host-only, zero device hours.

Read all nested snapshot events; verify unique consecutive step IDs, exactly one submission and completion each, explicit recorder errors/pending/truncation, and model/batch identity from source and collected logs. Summarize executed prefill lengths, active rows, sampling modes and perf_counter bounds. These are adapter-observed host intervals including host work and idle gaps, never device time. Do not infer request identities or emitted token counts. No source changes, launches or score audit.
