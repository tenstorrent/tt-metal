# Core subset write (experimental)

Restricts a sharded host-to-device write to a subset of the buffer's logical cores, using a `CoreRangeSet` filter.

Used for partial weight loads and similar staged transfers. May graduate to a stable API once the surface settles.

See `metal-api-policy.txt` at the repository root for experimental API conventions.
