# Earlier identity guard scheduling

This candidate changes only when UNPACK's scalar maximum comparison executes.
The source argument below was independently reviewed before its bounded
device smoke. Device success/performance must be read from evidence records;
this argument alone is not qualification.

1. QK phase row group j>0 first executes the preceding group's softmax. That
   call waits `cur.max` through j×2 tiles. Its maximum reduction previously
   waited the corresponding `prev.max` tiles. After issuing the first QK
   matmul for j, UNPACK compares those already-published maxima for j−1.
2. The final row group's softmax executes in the phase2 drain and waits its
   maximum before the first partial PV matmul. Immediately after issuing that
   first PV matmul, UNPACK compares the final group's maxima.
3. Neither maximum CB's front pointer nor its values change within this K
   step. The scalar flags are fresh stack-local values for this invocation,
   filled on UNPACK only. No scan runs in the first K chunk, where flags are
   unused because no previous state correction is required.
4. At the original correction point, UNPACK sends the stored flag through the
   original Math/Pack mailboxes. No mailbox is sent early. MATH/PACK obtain
   that received flag rather than trusting their unfilled local arrays.
5. All correction CB reserve/push/pop operations and publication fences remain
   at their original locations. The ordinary correction and the exact-one
   state replay are unchanged from the independently reviewed identity test.

The prototype statically requires Sqtiles8, QK/PV row height2 and materialized
V; the harness fixes Q256/K512/D128 and the original two input slots. No claim
is made for other geometries or arbitrary production features. The full
64-value finite-bit guard remains exact, not a checksum or sample.

The intended overlap is scalar UNPACK work during already-issued matmul and
packing. Whether any latency is hidden must be determined from fresh paired
timing against the v1 winner; compiler scheduling or upstream stalls may
erase the opportunity.
