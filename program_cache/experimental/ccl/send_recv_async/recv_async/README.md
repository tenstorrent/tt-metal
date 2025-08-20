# Program cache review — experimental/ccl/send_recv_async/recv_async

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra with override callback. Reader/writer kernels update output buffer address and fabric connection rt args on cache-hit.
- Hash includes tensor layout/shape/dtype/mem config and mesh socket properties via attributes; sufficient for program selection.

## References
- Program and override: `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/recv_async_program.cpp`.

## Notes
- If mesh socket topology or buffering policy evolves beyond what attributes capture, extend the hashed attributes accordingly.
