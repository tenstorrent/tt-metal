# Requests call10/call11 cap2048 extensions after V

Candidate only; no admission or execution. It contains exactly two U-root-reviewed inconclusive cap1024 cases: served Requests5414 call10 seed42 and full Requests5414 call11 seed42. Both retain the exact original 2048-token prompt IDs, chosen continuation, sampling parameters, and seed from their cap1024 fixtures. Each requests 2048 free-run tokens once with branches empty.

The forecast uses completed U cap1024 outputs: scale observed decode time for 1023 replay calls by the cumulative live-context sum ratio to the cap-safe 2047 replay calls. Because the prompt is unchanged, it preserves measured source prefill/first-token host overhead. It adds measured same-mode model load and one first generator-trace preparation proxy per mode. Estimates are planning bounds, not performance claims.

U review is now closed: full Requests call11 seed43 is classified `other` for a confirmed stray own token (correction recorded separately, without mutating the original pinned N record); full call10 seeds43 and44 are parser-confirmed `missing_tool`. Those rows are excluded from this cap-extension candidate.

The inert serial driver waits for explicit admission plus V suffix4096 successful full completion and terminal queue status, checks pinned hashes and device owners before each batch, and stops on timeout or nonzero exit.
