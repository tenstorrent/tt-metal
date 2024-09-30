# TT-Metal

TT-Metal is a low level programming model with user facing [host](./host_api.hpp) APIs.  This api did not evolve at the pace of the changes needed within metal and it has become clear that we need an overall to support our long term goals.


## Current API Limitations

- **Inconsistent interfaces and return types**
- **Clients access internal structures directly** (over 100 instances)
- **Lack of standardized query/set methods**
- **Confusing CoreCoord implementation**
- **Ownership and lifetime of buffers inconsistent**

### Impact:
- **Difficult for clients to use and maintain**
- **Increases support and development overhead**
- **Hinders future scalability and feature additions**

## Schedule for Delivering Work

### Overview

As we prepare for release 1.0, it is critical to stabilize the APIs to ensure consistency and maintainability across the codebase. The following changes are required:

- **Consistent Type Names**: We need to standardize type names across the API to avoid confusion and improve clarity for users.
- **Opaque Types**: To ensure better encapsulation and modularity, types should be made opaque. This will prevent external code from directly accessing or relying on internal details.
- **API Consolidation**: We need to flesh out the APIs so that internal objects and their methods are not used directly by external code. All interactions should occur through well-defined API methods, keeping the internal implementations isolated.


The work on these api changes will be on an independent branch: `metal-api-1.0` with rebasing on main done weekly.

Below is the schedule for implementing these changes.

### Timeline

| Date/Range                     | Task                                              |
|--------------------------------|---------------------------------------------------|
| **September 30**               | Begin work on Opaque Types                        |
| **October 14**                 | Specify New API                                   |
| **October 18**                 | Complete Opaque Types, Specify New API            |
| **October 14 - 25**            | Plan Complex API Changes & Notify, Review New API |
| **October 22 - November 1**   | Implement New API                                 |
| **November 1**                 | Merge to Main and Tag API Release 0.7             |
| **October 28 - November 8**    | Remove Reacharounds                               |
| **November 8 - November 13**   | Review Complex API Changes                        |
| **November 16 - November 21**  | Implement Complex API Changes                     |
| **November 21**                | Merge to Main and Tag API Release 1.0             |
| **December 5**                 | Delete namespace V0                               |
