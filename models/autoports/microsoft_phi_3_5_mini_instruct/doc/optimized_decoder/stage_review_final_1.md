# Stage review: final pass 1

Verdict: **more-work-needed**

The reviewer found two P2 issues:

1. The final batch-32 prefill profile spent 3.955 ms (20.87% of device
   time) in six manual-RoPE permutes. Global fused RoPE had been rejected for
   recorded-cache decode correctness, but a phase-specific fused-prefill and
   manual-decode adaptation had not been measured.
2. The four `tracy_final/*_perf_report.txt` files held CSV command diagnostics
   instead of human-readable advice tables. The matching console logs were
   empty, and README descriptions of the prefill block widths, functional
   reproduction environment, and default SDPA policy were stale.

Resolution:

- Implemented and measured a separate pair-basis prefill QKV/table path with a
  device-only canonical-cache adapter. It passes prefill and cache-consuming
  decode PCC at b1/b32 but regresses warmed prefill to 1.626/28.358 ms, so the
  selected default remains manual RoPE and the candidate is opt-in only.
- Recorded the candidate's exact persistent allocation and kept the selected
  context contract unchanged.
- Regenerated all four human-readable reports and machine-readable CSVs,
  populated console logs, and corrected the README claims and commands.

The review also noted non-blocking residual risk: semantic nonzero input near
the exact 131072-token prefill limit is not affordable in this layer gate, and
the functional prefill comparison uses synthetic values while optimized
prefill uses recorded activations. Both limitations are disclosed.
