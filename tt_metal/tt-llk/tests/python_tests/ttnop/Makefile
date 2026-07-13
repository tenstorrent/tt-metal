CXX      ?= g++
# -fopenmp: the `batch` sub-command generates the perturbed ELF sets in parallel.
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fopenmp
PREFIX   ?= $(HOME)/.local

ttnop: ttnop.cpp include/util.h
	$(CXX) $(CXXFLAGS) -o $@ ttnop.cpp

.PHONY: test clean install
test: ttnop
	./tests/run.sh

clean:
	rm -f ttnop
	rm -rf tests/work

install: ttnop
	install -Dm755 ttnop $(PREFIX)/bin/ttnop
