// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef TRISC_MATH
#define MAIN math_main()
#endif

#ifdef TRISC_PACK
#define MAIN pack_main()
#endif

#ifdef TRISC_UNPACK
#define MAIN unpack_main()
#endif
