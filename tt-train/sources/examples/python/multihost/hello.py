# hello.py
from mpi4py import MPI
import os
import socket


def main():
    hostname = socket.gethostname()
    rank = MPI.COMM_WORLD.Get_rank()
    print("Hello. I am %s, rank %i" % (hostname, rank))


if __name__ == "__main__":
    main()
