// Multithreaded warm-incremental CaDiCaL enumeration: W workers, each its own solver + seed, sharing LEARNED clauses
// (speed) and BLOCKING clauses (distinctness). Optional phase HINT: hint half the workers toward the latest found
// model (exploit) while the rest stay un-hinted (explore). Tests whether 16-way diversity kills the single-thread
// #2 phase-thrash. Usage: mt_warm_enum <cnf> <N> <workers> <hint:0|1>
#include <cadical.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace std::chrono;

struct Pool {
    std::mutex m;
    std::vector<std::vector<int>> cl;
    std::vector<int> prod;
    void publish(int pid, const std::vector<int>& c) { std::lock_guard<std::mutex> lk(m); cl.push_back(c); prod.push_back(pid); }
    void drain(int cons, size_t& cur, std::vector<std::vector<int>>& out) {
        std::lock_guard<std::mutex> lk(m);
        for (size_t i = cur; i < cl.size(); ++i) if (prod[i] != cons) out.push_back(cl[i]);
        cur = cl.size();
    }
    size_t size() { std::lock_guard<std::mutex> lk(m); return cl.size(); }
};

struct Learner : CaDiCaL::Learner {
    Pool* pool; int pid, maxsz; std::vector<int> buf;
    Learner(Pool* p, int id, int m) : pool(p), pid(id), maxsz(m) {}
    bool learning(int size) override { return size > 0 && size <= maxsz; }
    void learn(int lit) override { if (lit) buf.push_back(lit); else if (!buf.empty()) { pool->publish(pid, buf); buf.clear(); } }
};

std::mutex g_mtx;
std::unordered_set<size_t> g_seen;
std::atomic<int> g_found{0};
int g_target;
Pool g_learn, g_block;
steady_clock::time_point g_t0;
std::vector<double> g_times;

static size_t hashv(const std::vector<int>& m) { size_t h = 1469598103934665603ULL; for (int x : m) { h ^= (size_t)x; h *= 1099511628211ULL; } return h; }

void worker(const char* cnf, int wid, bool hint) {
    CaDiCaL::Solver s; s.set("quiet", 1); s.set("ilb", 2); s.set("seed", wid);
    Learner ln(&g_learn, wid, 8); s.connect_learner(&ln);
    int vars = 0; s.read_dimacs(cnf, vars, 1);
    size_t lcur = 0, bcur = 0; std::vector<std::vector<int>> imp;
    for (;;) {
        if (g_found.load() >= g_target) return;
        imp.clear(); g_block.drain(wid, bcur, imp); for (auto& c : imp) { for (int l : c) s.add(l); s.add(0); }
        imp.clear(); g_learn.drain(wid, lcur, imp); for (auto& c : imp) { for (int l : c) s.add(l); s.add(0); }
        s.limit("conflicts", 20000); int r = s.solve(); s.limit("conflicts", -1);
        if (r == 20) return;      // exhausted
        if (r != 10) continue;    // budget window -> re-import, retry
        std::vector<int> model(vars); for (int v = 1; v <= vars; v++) model[v - 1] = s.val(v);
        const size_t h = hashv(model);
        bool rec = false;
        { std::lock_guard<std::mutex> lk(g_mtx);
          if (g_found.load() < g_target && g_seen.find(h) == g_seen.end()) {
              g_seen.insert(h); g_found.fetch_add(1); rec = true;
              g_times.push_back(duration<double>(steady_clock::now() - g_t0).count());
          } }
        std::vector<int> blk; blk.reserve(vars);
        for (int v = 1; v <= vars; v++) blk.push_back(model[v - 1] > 0 ? -v : v);
        if (rec) g_block.publish(wid, blk);
        for (int l : blk) s.add(l); s.add(0);   // block it in my own solver
        if (hint) for (int v = 1; v <= vars; v++) s.phase(model[v - 1]);  // exploit: bias toward this region
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s cnf N [workers] [hint0/1]\n", argv[0]); return 1; }
    const char* cnf = argv[1]; g_target = atoi(argv[2]);
    const int W = argc > 3 ? atoi(argv[3]) : 16;
    const int hintmode = argc > 4 ? atoi(argv[4]) : 0;
    const int seedbase = argc > 5 ? atoi(argv[5]) : 0;  // shift the 16 worker seeds to sample a different seed set
    g_t0 = steady_clock::now();
    std::vector<std::thread> ts;
    for (int w = 0; w < W; w++) ts.emplace_back(worker, cnf, seedbase + w + 1, hintmode && (w % 2 == 0));  // half exploit
    for (auto& t : ts) t.join();
    double prev = 0;
    for (size_t i = 0; i < g_times.size(); i++) { printf("mt sol %zu: %.2fs (delta %.2fs)\n", i + 1, g_times[i], g_times[i] - prev); prev = g_times[i]; }
    printf("mt TOTAL: %.2fs  (%d workers, hint=%d, learn_pool=%zu, block_pool=%zu)\n",
           duration<double>(steady_clock::now() - g_t0).count(), W, hintmode, g_learn.size(), g_block.size());
    return 0;
}
