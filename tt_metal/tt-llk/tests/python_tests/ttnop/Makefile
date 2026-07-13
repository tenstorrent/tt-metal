CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
PREFIX   ?= $(HOME)/.local

ttnop: ttnop.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: test clean install
test: ttnop
	./tests/run.sh

clean:
	rm -f ttnop
	rm -rf tests/work

install: ttnop
	install -Dm755 ttnop $(PREFIX)/bin/ttnop
