// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mpi.h>

// mpi-ext.h declares MPIX_* when Open MPI is built with ULFM; keep this in a
// header so every TU sees the same probe and unity builds do not redefine
// OMPI_HAS_ULFM in multiple .cpp chunks.
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

#if !defined(OMPI_HAS_ULFM)
#if (defined(OPEN_MPI) && OPEN_MPI && defined(MPIX_ERR_PROC_FAILED))
#define OMPI_HAS_ULFM 1
#else
#define OMPI_HAS_ULFM 0
#endif
#endif
