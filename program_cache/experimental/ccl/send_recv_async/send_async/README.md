# Program cache review — experimental/ccl/send_recv_async/send_async

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra with override callback. Reader updates input buffer address; writer updates fabric connection rt args.
- Hash keys include tensor properties and mesh socket attributes used in compile-time args.

## References
- Program and override: `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/send_async_program.cpp`.

## Notes
- Keep arg ordering consistent when adding runtime args; centralize indices to avoid override drift.
