Point-to-Point program cache review

Status: Reviewed — no program cache issues found.

Summary
- Factory: `device/host/point_to_point_device_op.cpp` (mesh device operation with typed ProgramFactory).
- Override updates per-coordinate buffer addresses and semaphore address for both sender and receiver programs:
  - Sender: reader arg[0] = input mesh buffer addr; writer arg[0] = intermediate output addr; writer arg[8] = semaphore addr.
  - Receiver: reader arg[3] = intermediate input addr; reader arg[7] = semaphore addr; writer arg[0] = final output addr.
- Program hash path uses mesh workload hashing that includes tensor coordinates, preventing collisions across placements.

Recommendation
- No changes required.
