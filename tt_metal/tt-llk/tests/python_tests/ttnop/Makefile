CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
OPENMP   := -fopenmp
PREFIX   ?= $(HOME)/.local

ttnop: ttnop.cpp include/util.h
	$(CXX) $(CXXFLAGS) $(OPENMP) -o $@ ttnop.cpp

.PHONY: test clean install
test: ttnop
	./tests/run.sh

clean:
	rm -f ttnop
	rm -rf tests/work

install: ttnop
	install -Dm755 ttnop $(PREFIX)/bin/ttnop
