# tt::tt_metal::internal

**For Tenstorrent-internal use. Unstable. Incomplete. Subject to frequent change.**

This directory exposes APIs under the `tt::tt_metal::internal` namespace that are used across
internal components of tt-metal but are **not part of the public contract**. External users and
downstream consumers should avoid depending on anything here - headers, symbols, and interfaces may
be added, removed, or restructured at any time.

## How this differs from `tt-metalium/experimental`

| | `tt-metalium/experimental` | `api/internal` |
|---|---|---|
| **Audience** | External developers willing to accept churn | Internal Tenstorrent teams only |
| **Intent** | In-development features on a path to productization into the stable `tt-metalium` API | Cross-cutting infrastructure with no commitment to stabilization |
| **Stability** | Unstable but intentionally designed for eventual promotion | No stability guarantee; may never be promoted |
| **Use at your own risk?** | Yes, but changes are documented | Yes, and changes may be silent |

In short: `experimental` is the public backdoor for power users who want early access to features
that are heading toward the stable API. `internal` is not intended for external consumption.
