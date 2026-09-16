// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/* tharns.h -- ttas test harness
 * Ripped from Takahe (which ripped it from BarraCUDA) with love and
 * no remorse. The takahe is flightless; so is this assembler until it
 * learns to encode. */
#ifndef THARNS_H
#define THARNS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

typedef void (*tfunc_t)(void);

typedef struct {
    const char *tname;
    const char *tcats;
    tfunc_t     func;
} tcase_t;

#define TH_MAXTS 512
#define TH_BUFSZ 65536

extern tcase_t th_list[];
extern int th_cnt;
extern int npass, nfail, nskip;

/* ---- Self-Registration ----
 * gcc/clang constructor attribute auto-registers tests.
 * Like a self-seating restaurant: you sit where you like
 * and the system figures out who you are. */
#define TH_REG(cat, fn) \
    __attribute__((constructor)) static void reg_##fn(void) { \
        if (th_cnt < TH_MAXTS) \
            th_list[th_cnt++] = (tcase_t){#fn, cat, fn}; \
    }

/* ---- Assertions ----
 * The test stops here. No appeals. No remand. */
#define CHECK(x) do { if (!(x)) { \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    nfail++; return; } } while(0)

#define CHEQ(a, b)   CHECK((a) == (b))
#define CHNE(a, b)   CHECK((a) != (b))
#define CHSTR(a, b)  CHECK(strcmp((a),(b)) == 0)
#define PASS()       do { npass++; } while(0)
#define SKIP(r)      do { nskip++; printf("  SKIP: %s\n", r); return; } while(0)

/* ---- Binary Path ---- */
#ifdef _WIN32
#define TTAS_BIN ".\\ttas.exe"
#else
#define TTAS_BIN "./ttas"
#endif

/* ---- Utilities ---- */
int th_run(const char *cmd, char *obuf, int osz);
int th_exist(const char *path);

#endif /* THARNS_H */
