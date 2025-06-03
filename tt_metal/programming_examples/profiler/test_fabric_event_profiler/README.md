### programming example - experimental profiler tt-fabric tracing

**NOTE**: This feature is incomplete, experimental, and under active development.

This test shows how to instrument a data movement kernel
(`test_fabric_event_profiler/kernels/tt_fabric_1d_tx.cpp`) with calls to
`RECORD_FABRIC_EVENT()`, such that data sent to EDM routers can be properly
traced and coalesced into logical fabric packet events by the device profiler.

```cpp
    // RECORD_FABRIC_HEADER is defined in fabric_event_profiler.hpp
    #include "tt_metal/tools/profiler/experimental/fabric_event_profiler.hpp"
    ...
    // the fabric packet header must be recorded by calling RECORD_FABRIC_HEADER
    // *before* the fabric payload is sent to EDM router
    RECORD_FABRIC_HEADER(packet_header);
    connection.send_payload_without_header_non_blocking_from_address(...);
    connection.send_payload_blocking_from_address(...);
```
