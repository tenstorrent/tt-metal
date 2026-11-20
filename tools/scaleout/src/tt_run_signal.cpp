// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt-run-signal: send a signal to every rank of a running prterun job, on every host, through PRRTE job control.
//
//   tt-run-signal --uri <prterun PMIx server URI> --signal <N>
//
// tt-run uses it to stop a job: PMIx_Job_control(PMIX_JOB_CTRL_SIGNAL = SIGTERM) makes prterun tell every daemon to
// kill() its own ranks with that signal. Unlike a signal sent to prterun itself (SIGTERM, then SIGKILL 250 ms
// later), the job is not aborted and no SIGKILL follows. The URI is the first line of prterun's tool rendezvous file
// (<tmpdir>/prterun.<host>.<pid>.<uid>/pmix.<host>.tool.<pid>); connecting by URI avoids PMIx's pid lookup, which
// searches the whole system tmpdir recursively. Exit code: 0 on success, 1 on a PMIx error, 2 on bad arguments.

#include <pmix_tool.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>  // getpid: POSIX, no C++ header equivalent

int main(int argc, char** argv) {
    const char* uri = nullptr;
    int32_t signum = 15;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--uri") == 0) {
            uri = argv[i + 1];
        } else if (std::strcmp(argv[i], "--signal") == 0) {
            signum = static_cast<int32_t>(std::atoi(argv[i + 1]));
        }
    }
    if (uri == nullptr) {
        std::fprintf(stderr, "usage: %s --uri <prterun PMIx URI> --signal <N>\n", argv[0]);
        return 2;
    }

    // Identify as a tool with a namespace of our own, the way PRRTE's own tools (pterm) do.
    char nspace[PMIX_MAX_NSLEN + 1];
    std::snprintf(nspace, sizeof nspace, "tt-run-signal.%d", static_cast<int>(getpid()));
    pmix_rank_t rank = 0;
    pmix_info_t init[3];
    PMIX_INFO_LOAD(&init[0], PMIX_TOOL_NSPACE, nspace, PMIX_STRING);
    PMIX_INFO_LOAD(&init[1], PMIX_TOOL_RANK, &rank, PMIX_PROC_RANK);
    PMIX_INFO_LOAD(&init[2], PMIX_SERVER_URI, uri, PMIX_STRING);
    pmix_proc_t me;
    pmix_status_t rc = PMIx_tool_init(&me, init, 3);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "tt-run-signal: PMIx_tool_init(%s): %s\n", uri, PMIx_Error_string(rc));
        return 1;
    }

    // No targets: every rank of the job this prterun runs.
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    rc = PMIx_Job_control(nullptr, 0, &directive, 1, nullptr, nullptr);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "tt-run-signal: PMIx_Job_control(signal %d): %s\n", signum, PMIx_Error_string(rc));
    }
    PMIx_tool_finalize();
    return rc == PMIX_SUCCESS ? 0 : 1;
}
