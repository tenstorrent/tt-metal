// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

/* tmain.c -- tt-mop test runner.
 * The expander is the load-bearing wall of the whole optimiser, so its
 * tests run first and run loud. */

#include "tharns.h"

tcase_t th_list[TH_MAXTS];
int th_cnt = 0;
int npass  = 0;
int nfail  = 0;
int nskip  = 0;

int th_run(const char *cmd, char *obuf, int osz)
{
    char full[TH_BUFSZ];
    snprintf(full, TH_BUFSZ, "%s 2>&1", cmd);
    FILE *fp = popen(full, "r");
    if (!fp) { obuf[0] = '\0'; return -1; }
    int n = (int)fread(obuf, 1, (size_t)(osz - 1), fp);
    if (n < 0) n = 0;
    obuf[n] = '\0';
    int rc = pclose(fp);
#ifndef _WIN32
    if (rc != -1 && (rc & 0xFF) == 0)
        rc = (rc >> 8) & 0xFF;
#endif
    return rc;
}

int th_exist(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

int main(int argc, char **argv)
{
    int i;
    const char *filter = NULL;

    if (argc > 1 && strcmp(argv[1], "--all") != 0)
        filter = argv[1];

    printf("\ntt-mop Test Suite\n");
    printf("=================\n");

    for (i = 0; i < th_cnt; i++) {
        if (filter && strcmp(filter, th_list[i].tcats) != 0)
            continue;

        int before = npass + nfail + nskip;
        printf("  %-28s", th_list[i].tname);
        fflush(stdout);

        th_list[i].func();

        int after = npass + nfail + nskip;
        if (after == before)
            npass++;            /* no verdict means it didn't crash: pass */
        if (nfail == 0 || after > before)
            printf("PASS\n");
    }

    printf("=================\n");
    printf("%d tests: %d passed, %d failed, %d skipped\n\n",
           npass + nfail + nskip, npass, nfail, nskip);

    return nfail > 0 ? 1 : 0;
}
