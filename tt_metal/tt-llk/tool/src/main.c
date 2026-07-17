// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* main.c -- tt-mop driver.
 *
 * The front desk. It takes a flat binary of 32-bit Tensix words, the
 * sort ttas spits out, and points the optimiser at it: --opt reports the
 * cheaper plan and what it bought you, --suggest prints a do-this-by-
 * offset replay arrangement a human can actually apply, and --demo
 * expands a sample plan so you can watch the expander earn its keep. Not
 * much cleverness lives here. It all lives one door down.
 */

#include "ttmop.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr,
        "tt-mop -- open Tensix MOP/replay optimiser\n"
        "usage:\n"
        "  ttmop --demo            expand a sample MOP plan, print the stream\n"
        "  ttmop --dump FILE.bin   print a flat little-endian word file\n"
        "  ttmop --opt  FILE.bin [slots]  optimise a stream, report the saving\n"
        "                          (slots = replay buffer this thread may use)\n"
        "  ttmop --suggest FILE.bin [slots]  print an applicable replay suggestion\n"
        "  ttmop --help            this message\n");
}

/* Expand a small double-loop plan and print the words it produces.
 * Shows the expander doing its one job, before any search exists. */
static int run_demo(void)
{
    static uint8_t   arena_buf[1u << 20];
    ka_arena_t       A;
    tm_planop_t      plan[2];
    tm_stream_t      s;

    memset(plan, 0, sizeof(plan));
    ka_init(&A, arena_buf, (uint32_t)sizeof(arena_buf), 0);

    /* A two-by-two double loop in the matmul's shape: a start op, then an
     * inner body of one op per step (alternating between two values here,
     * since loop_op1 is non-NOP, which doubles the inner count) with the
     * last step swapped out for loop1_last / loop0_last, and a pair of
     * end ops. */
    plan[0].kind             = TM_CFG_DOUBLE;
    plan[0].dtmpl.outer_len  = 2u;
    plan[0].dtmpl.inner_len  = 2u;
    plan[0].dtmpl.start_op0  = 0xAA000001u;
    plan[0].dtmpl.loop_op0   = 0xBB000002u;
    plan[0].dtmpl.loop_op1   = 0xCC000003u;
    plan[0].dtmpl.loop1_last = 0xDD000004u;
    plan[0].dtmpl.loop0_last = 0xEE000005u;
    plan[0].dtmpl.end_op0    = 0xFF000006u;
    plan[0].dtmpl.end_op1    = 0x11000007u;
    plan[1].kind             = TM_MOP_RUN;

    s = tm_expand(plan, 2u, &A);
    if (!s.ok) {
        fprintf(stderr, "tt-mop: expand failed: %s\n",
                s.err ? s.err : "?");
        return 1;
    }

    printf("MOP double-loop plan expands to %u Tensix words:\n", (unsigned)s.n);
    for (uint32_t i = 0u; i < s.n; i++) {
        printf("  [%2u] 0x%08X  (opcode 0x%02X)\n",
               (unsigned)i, (unsigned)s.words[i],
               (unsigned)(s.words[i] >> TM_OP_SHIFT));
    }
    return 0;
}

/* Read a flat little-endian binary of 32-bit words and print them.
 * Bounded by TM_MAX_WORDS; anything past that is reported and ignored. */
static int run_dump(const char *path)
{
    FILE     *f;
    uint32_t  count = 0u;
    uint8_t   b[4];

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "tt-mop: cannot open '%s'\n", path);
        return 1;
    }

    printf("%s:\n", path);
    while (count < TM_MAX_WORDS && fread(b, 1u, 4u, f) == 4u) {
        tm_word_t w = (tm_word_t)b[0]
                    | ((tm_word_t)b[1] << 8)
                    | ((tm_word_t)b[2] << 16)
                    | ((tm_word_t)b[3] << 24);
        printf("  [%4u] 0x%08X  (opcode 0x%02X)\n",
               (unsigned)count, (unsigned)w,
               (unsigned)(w >> TM_OP_SHIFT));
        count++;
    }
    fclose(f);
    printf("%u words.\n", (unsigned)count);
    return 0;
}

/* Read a flat little-endian word file, optimise it, and report what
 * the optimiser managed to shave off and how. max_slots is the replay
 * buffer this thread may use (the bounty's sharing constraint). */
static int run_opt(const char *path, uint32_t max_slots)
{
    static tm_word_t   words[TM_MAX_WORDS];
    static uint8_t     arena_buf[1u << 21];   /* 2 MiB: plan + verify scratch */
    ka_arena_t         A;
    tm_optresult_t     r;
    FILE              *f;
    uint32_t           n = 0u;
    uint8_t            b[4];

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "tt-mop: cannot open '%s'\n", path);
        return 1;
    }
    while (n < TM_MAX_WORDS && fread(b, 1u, 4u, f) == 4u) {
        words[n] = (tm_word_t)b[0]
                 | ((tm_word_t)b[1] << 8)
                 | ((tm_word_t)b[2] << 16)
                 | ((tm_word_t)b[3] << 24);
        n++;
    }
    fclose(f);

    ka_init(&A, arena_buf, (uint32_t)sizeof(arena_buf), 0);
    r = tm_optimise_budget(words, n, max_slots, &A);

    printf("tt-mop: %s\n", path);
    printf("  input          : %u Tensix words\n", (unsigned)n);
    printf("  buffer budget  : %u slots\n", (unsigned)max_slots);
    printf("  pass           : %s\n", r.note ? r.note : "?");
    printf("  verified       : %s\n", r.verified ? "yes" : "NO");
    printf("  RISCV issues   : %u -> %u", (unsigned)r.cost_naive,
           (unsigned)r.cost_opt);
    if (r.cost_naive > r.cost_opt) {
        unsigned saved = (unsigned)(r.cost_naive - r.cost_opt);
        double pct = r.cost_naive ? (100.0 * (double)saved /
                                     (double)r.cost_naive) : 0.0;
        printf("   (saved %u, %.1f%%)\n", saved, pct);
    } else {
        printf("   (no saving)\n");
    }
    printf("  plan           : %u ops\n", (unsigned)r.n_ops);
    return r.verified ? 0 : 8;
}

/* Read a stream, optimise it, and print an applicable suggestion: the
 * replay arrangement, by source offset, that a maintainer can act on.
 * This is the deliverable shape the bounty asks for ("filter out the
 * good suggestions and apply them"). Direct instructions stay as they
 * are; only the replay actions are spelled out. */
static int run_suggest(const char *path, uint32_t max_slots)
{
    static tm_word_t   words[TM_MAX_WORDS];
    static uint8_t     arena_buf[1u << 21];
    ka_arena_t         A;
    tm_optresult_t     r;
    FILE              *f;
    uint32_t           n = 0u;
    uint8_t            b[4];
    uint32_t           src = 0u;
    uint32_t           n_blocks = 0u;
    uint32_t           n_direct = 0u;

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "tt-mop: cannot open '%s'\n", path);
        return 1;
    }
    while (n < TM_MAX_WORDS && fread(b, 1u, 4u, f) == 4u) {
        words[n] = (tm_word_t)b[0]
                 | ((tm_word_t)b[1] << 8)
                 | ((tm_word_t)b[2] << 16)
                 | ((tm_word_t)b[3] << 24);
        n++;
    }
    fclose(f);

    ka_init(&A, arena_buf, (uint32_t)sizeof(arena_buf), 0);
    r = tm_optimise_budget(words, n, max_slots, &A);

    printf("tt-mop suggestion for %s\n", path);
    printf("  %u Tensix words, buffer budget %u slots\n",
           (unsigned)n, (unsigned)max_slots);
    printf("  RISC-V issues %u -> %u%s, %s\n",
           (unsigned)r.cost_naive, (unsigned)r.cost_opt,
           r.verified ? "" : "  (UNVERIFIED)",
           r.verified ? "verified equivalent" : "NOT verified");

    if (r.cost_opt >= r.cost_naive) {
        printf("  no replay arrangement beats issuing directly.\n");
        return 0;
    }

    printf("\n  Apply this:\n");
    for (uint32_t k = 0u; k < r.n_ops; k++) {
        const tm_planop_t *op = &r.plan[k];
        switch (op->kind) {
        case TM_REPLAY_LOAD:
            n_blocks++;
            printf("  - record the %u instructions at source [%u..%u] into "
                   "replay slots [%u..%u), executing as they load\n",
                   (unsigned)op->rp_len, (unsigned)src,
                   (unsigned)(src + op->rp_len - 1u),
                   (unsigned)op->rp_start,
                   (unsigned)(op->rp_start + op->rp_len));
            src += op->rp_len;
            break;
        case TM_REPLAY_RUN:
            printf("  - replay slots [%u..%u) in place of source [%u..%u] "
                   "(one REPLAY instead of %u issues)\n",
                   (unsigned)op->rp_start,
                   (unsigned)(op->rp_start + op->rp_len),
                   (unsigned)src, (unsigned)(src + op->rp_len - 1u),
                   (unsigned)op->rp_len);
            src += op->rp_len;
            break;
        case TM_EMIT:
            n_direct++;
            src += 1u;
            break;
        case TM_CFG_DOUBLE:
        case TM_CFG_UNPACK:
        case TM_MOP_RUN:
        default:
            break;
        }
    }
    printf("\n  %u replay block(s), %u instructions left direct.\n",
           (unsigned)n_blocks, (unsigned)n_direct);
    return r.verified ? 0 : 8;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage();
        return 0;
    }
    if (strcmp(argv[1], "--demo") == 0) {
        return run_demo();
    }
    if (strcmp(argv[1], "--dump") == 0) {
        if (argc < 3) {
            fprintf(stderr, "tt-mop: --dump needs a file\n");
            return 1;
        }
        return run_dump(argv[2]);
    }
    if (strcmp(argv[1], "--opt") == 0) {
        uint32_t slots = TM_REPLAY_SLOTS;
        if (argc < 3) {
            fprintf(stderr, "tt-mop: --opt needs a file\n");
            return 1;
        }
        if (argc > 3) {
            long s = strtol(argv[3], NULL, 10);
            if (s > 0 && s <= (long)TM_REPLAY_SLOTS) {
                slots = (uint32_t)s;
            }
        }
        return run_opt(argv[2], slots);
    }
    if (strcmp(argv[1], "--suggest") == 0) {
        uint32_t slots = TM_REPLAY_SLOTS;
        if (argc < 3) {
            fprintf(stderr, "tt-mop: --suggest needs a file\n");
            return 1;
        }
        if (argc > 3) {
            long s = strtol(argv[3], NULL, 10);
            if (s > 0 && s <= (long)TM_REPLAY_SLOTS) {
                slots = (uint32_t)s;
            }
        }
        return run_suggest(argv[2], slots);
    }
    fprintf(stderr, "tt-mop: unknown argument '%s'\n", argv[1]);
    usage();
    return 1;
}
