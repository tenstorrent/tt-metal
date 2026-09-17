// SPDX-License-Identifier: Apache-2.0
//
// Clean public include for the vendored kissat SAT engine.
//
// This wrapper exists so consumers get ONLY the kissat public C API on their include path — NOT kissat's
// src/ directory. kissat ships its own internal src/limits.h (solver-limit structs, not the C standard
// <limits.h>), so putting src/ on a consumer's -I path shadows the system <limits.h> and breaks any
// consumer TU that (directly or transitively) includes <limits.h>/<climits> (e.g. UINT_MAX becomes
// undeclared). Exposing this include/ dir instead keeps kissat's src/ off the consumer -I path.
//
// upstream kissat.h has no `extern "C"` guard, so we add one here for C++ callers.
#ifndef TT_METAL_FABRIC_KISSAT_PUBLIC_H
#define TT_METAL_FABRIC_KISSAT_PUBLIC_H

#ifdef __cplusplus
extern "C" {
#endif

// Resolved relative to THIS file (include/), so kissat's src/ never lands on the consumer's -I path.
// kissat.h itself includes nothing, so this pulls in no siblings.
#include "../src/kissat.h"

#ifdef __cplusplus
}
#endif

#endif  // TT_METAL_FABRIC_KISSAT_PUBLIC_H
