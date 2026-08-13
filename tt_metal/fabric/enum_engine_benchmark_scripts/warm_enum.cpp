// Warm incremental-CaDiCaL enumeration: read a DIMACS CNF, then solve -> block full model -> solve, N times.
// One solver kept warm across solutions (learned clauses + phases persist). Prints per-solution wall time.
#include <cadical.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s cnf N\n", argv[0]); return 1; }
    const char* path = argv[1];
    const int N = atoi(argv[2]);
    const int seed = (argc > 3) ? atoi(argv[3]) : 0;
    CaDiCaL::Solver s;
    s.set("quiet", 1);
    s.set("ilb", 2);  // incremental lazy backtracking -- the enumeration tuning our solver uses
    s.set("seed", seed);
    int vars = 0;
    const char* err = s.read_dimacs(path, vars, 1);
    if (err) { fprintf(stderr, "read error: %s\n", err); return 1; }
    printf("warm: %d vars loaded\n", vars);
    const auto t0 = steady_clock::now();
    double prev = 0;
    for (int i = 1; i <= N; i++) {
        const int r = s.solve();
        const double el = duration<double>(steady_clock::now() - t0).count();
        if (r != 10) { printf("warm sol %d: NOT SAT (r=%d) at %.2fs\n", i, r, el); break; }
        printf("warm sol %d: %.2fs (delta %.2fs)\n", i, el, el - prev);
        fflush(stdout);
        prev = el;
        // Extract the FULL model first (adding a clause drops the solver out of the satisfied state, after which
        // val() is invalid), THEN add the blocking clause = negation of the model.
        std::vector<int> model(vars);
        for (int v = 1; v <= vars; v++) model[v - 1] = s.val(v);
        for (int v = 1; v <= vars; v++) s.add(model[v - 1] > 0 ? -v : v);
        s.add(0);
    }
    printf("warm TOTAL: %.2fs\n", duration<double>(steady_clock::now() - t0).count());
    return 0;
}
