// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// All internal detail classes (GraphIndexData, ConstraintIndexData, DFSSearchEngine, MappingValidator, etc.)
// are now defined in the public header topology_solver.hpp in the detail namespace.
// This file just includes the public header and then includes the template implementations.
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

// Include template implementations
#include "tt_metal/fabric/topology_solver_internal.tpp"
