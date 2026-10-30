// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t c) {
  volatile std::uint32_t* a = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
  volatile std::uint32_t* b = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE + 4);
  if (c) { TT_SETC16(1, a[0]); } else { TT_SETC16(1, b[0]); } }
void f(const std::uint32_t c) {
  volatile std::uint32_t* a = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
  volatile std::uint32_t* b = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE + 4);
  if (!c) { TT_SETC16(1, b[0]); } else { TT_SETC16(1, a[0]); } }
// same register read, different register's value used
void reference_g() {
  volatile std::uint32_t* a = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
  std::uint32_t x = a[0]; std::uint32_t y = a[1]; TT_SETC16(1, x); (void)y; }
void g() {
  volatile std::uint32_t* a = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
  std::uint32_t x = a[0]; std::uint32_t y = a[1]; TT_SETC16(1, y); (void)x; }
