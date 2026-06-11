// Host-side TCP throughput sink: accept one connection, drain it, report rate.
// usage: ./netsink <port>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int port = argc > 1 ? atoi(argv[1]) : 9000;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(ls, (void*)&a, sizeof a) || listen(ls, 1)) {
        perror("bind/listen");
        return 1;
    }
    printf("netsink listening on %d\n", port);
    fflush(stdout);
    int c = accept(ls, 0, 0);
    char* buf = malloc(1 << 20);
    long total = 0;
    int first = 1;
    struct timespec t0, t1;
    ssize_t n;
    while ((n = recv(c, buf, 1 << 20, 0)) > 0) {
        if (first) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            first = 0;
        }
        total += n;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf(
        "recv %.1f MB in %.3f s = %.2f MB/s (%.0f Mbit/s)\n", total / 1e6, s, total / 1e6 / s, total * 8.0 / 1e6 / s);
    return 0;
}
