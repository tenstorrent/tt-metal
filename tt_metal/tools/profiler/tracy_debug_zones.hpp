// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tracy/Tracy.hpp>

// Tracy zones and their zone annotations, compiled in only with the --build-perf-debug build flag.
#if defined(TT_TRACY_DEBUG)
#define TTZoneScopedD ZoneScoped
#define TTZoneScopedND(name) ZoneScopedN(name)
#define TTZoneScopedCD(color) ZoneScopedC(color)
#define TTZoneScopedNCD(name, color) ZoneScopedNC(name, color)
#define TTZoneScopedSD(depth) ZoneScopedS(depth)
#define TTZoneScopedNSD(name, depth) ZoneScopedNS(name, depth)
#define TTZoneScopedCSD(color, depth) ZoneScopedCS(color, depth)
#define TTZoneScopedNCSD(name, color, depth) ZoneScopedNCS(name, color, depth)
#define TTZoneTextD(txt, size) ZoneText(txt, size)
#define TTZoneNameD(txt, size) ZoneName(txt, size)
#define TTZoneColorD(color) ZoneColor(color)
#define TTZoneValueD(value) ZoneValue(value)
#define TTZoneIsActiveD ZoneIsActive
#else
#define TTZoneScopedD
#define TTZoneScopedND(name)
#define TTZoneScopedCD(color)
#define TTZoneScopedNCD(name, color)
#define TTZoneScopedSD(depth)
#define TTZoneScopedNSD(name, depth)
#define TTZoneScopedCSD(color, depth)
#define TTZoneScopedNCSD(name, color, depth)
#define TTZoneTextD(txt, size)
#define TTZoneNameD(txt, size)
#define TTZoneColorD(color)
#define TTZoneValueD(value)
#define TTZoneIsActiveD false
#endif
