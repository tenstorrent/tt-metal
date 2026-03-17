# Deferred Items - Phase 01

## Pre-existing Build Error

**File:** `tt_metal/fabric/erisc_datamover_builder.cpp:1025`
**Error:** `no member named 'get_all_stream_ids' in 'tt::tt_fabric::StreamRegAssignments'`
**Context:** This error exists on the base branch prior to any 01-02 changes. It appears to be a consequence of the StreamRegAssignments restructuring from plan 01-01. Not in scope for 01-02.
