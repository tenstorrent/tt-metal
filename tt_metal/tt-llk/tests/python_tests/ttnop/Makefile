# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

scan-%:
	$(MAKE) --no-print-directory -C native $@

clean:
	$(MAKE) --no-print-directory -C native clean

check:
	python3 tests/ttnop_check.py

check-hang:
	python3 tests/ttnop_check.py --hang

.PHONY: clean check check-hang
