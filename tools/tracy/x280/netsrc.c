// X280-side TCP throughput source: send <MB> to <ip>:<port> as fast as possible.
// usage: ./netsrc <ip> <port> <MB>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <ip> <port> <MB>\n", argv[0]);
        return 1;
    }
    const char* ip = argv[1];
    int port = atoi(argv[2]);
    long total = atol(argv[3]) << 20;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, ip, &a.sin_addr);
    if (connect(s, (void*)&a, sizeof a)) {
        perror("connect");
        return 1;
    }
    char* buf = calloc(1, 1 << 20);
    long sent = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (sent < total) {
        ssize_t n = send(s, buf, 1 << 20, 0);
        if (n <= 0) {
            perror("send");
            break;
        }
        sent += n;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    close(s);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("sent %.1f MB in %.3f s = %.2f MB/s\n", sent / 1e6, sec, sent / 1e6 / sec);
    return 0;
}
