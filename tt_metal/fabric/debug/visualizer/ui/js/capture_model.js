// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Joins every loaded rank's decoded file into one model: meshes, chips placed by mesh_coord, routers by path, and the
// links between routers. Each chip comes from the rank where it's is_local; a chip whose rank isn't loaded keeps its
// coordinates and has no routers. Records point at the decoded JSON (`data`) rather than copying it.

export const DIRECTION_NAMES = { N: "North", E: "East", S: "South", W: "West", Z: "Z" };
const DIRECTION_ORDER = Object.keys(DIRECTION_NAMES);
const ROUTER_NAME_PATTERN = /^([NESWZ])(\d+)$/;

export function buildCaptureModel(loadedCapture) {
    const firstDecoded = loadedCapture.ranks[0].decoded;
    const problems = [];
    const meshes = new Map();
    const chipsByPath = new Map();
    const routersByPath = new Map();

    for (const rank of loadedCapture.ranks) {
        for (const [meshId, meshData] of Object.entries(rank.decoded.meshes ?? {})) {
            if (!meshes.has(meshId)) meshes.set(meshId, createMesh(meshId, meshData));
            const mesh = meshes.get(meshId);
            for (const [chipId, chipData] of Object.entries(meshData.chips ?? {})) {
                const chipPath = `${meshId}/${chipId}`;
                if (!chipsByPath.has(chipPath)) {
                    const chip = createChip(mesh, chipId, chipPath, chipData);
                    chipsByPath.set(chipPath, chip);
                    mesh.chips.push(chip);
                }
                const chip = chipsByPath.get(chipPath);
                if (!chipData.is_local) continue;
                if (chip.ownerRankNumber !== null) {
                    problems.push(`${chipPath} is local to both rank ${chip.ownerRankNumber} and rank ${rank.number}.`);
                    continue;
                }
                chip.data = chipData;
                chip.ownerRankNumber = rank.number;
                for (const [routerName, routerData] of Object.entries(chipData.routers ?? {})) {
                    const router = createRouter(chip, routerName, routerData, rank.number);
                    chip.routers.push(router);
                    routersByPath.set(router.path, router);
                }
            }
        }
    }

    for (const mesh of meshes.values()) placeChips(mesh, problems);
    for (const chip of chipsByPath.values()) groupRoutersByDirection(chip);
    const { links, linkByRouterPath } = buildLinks(routersByPath);

    return {
        capture: firstDecoded.capture,
        decode: firstDecoded.decode,
        fabric: firstDecoded.fabric,
        arch: firstDecoded.arch,
        vocabulary: firstDecoded.vocabulary,
        analysisVocabulary: firstDecoded.analysis?.vocabulary,
        worldSize: firstDecoded.run.world_size,
        ranks: loadedCapture.ranks,
        missingRankNumbers: loadedCapture.missingRankNumbers,
        loadWarnings: loadedCapture.warnings,
        meshes,
        chipsByPath,
        routersByPath,
        links,
        linkByRouterPath,
        coverage: sumCoverage(loadedCapture.ranks),
        discoverySkippedCores: findDiscoverySkippedRouters(loadedCapture.ranks, chipsByPath),
        findings: loadedCapture.ranks.flatMap((rank) =>
            (rank.decoded.analysis?.findings ?? []).map((finding) => ({ ...finding, rankNumber: rank.number }))),
        problems,
    };
}

function createMesh(meshId, meshData) {
    const [rows, columns] = meshData.shape;
    return { id: meshId, path: meshId, rows, columns, torus: meshData.torus, data: meshData, chips: [], chipGrid: [] };
}

function createChip(mesh, chipId, chipPath, chipData) {
    const [row, column] = chipData.mesh_coord;
    return {
        id: chipId,
        path: chipPath,
        mesh,
        row,
        column,
        hostRank: chipData.host_rank,
        // The is_local copy once a rank that owns the chip is loaded; until then, a copy with coordinates only.
        data: chipData,
        ownerRankNumber: null,
        routers: [],
        routersByDirection: new Map(),
    };
}

function createRouter(chip, routerName, routerData, rankNumber) {
    const nameMatch = ROUTER_NAME_PATTERN.exec(routerName);
    return {
        name: routerName,
        path: `${chip.path}/${routerName}`,
        chip,
        direction: nameMatch?.[1] ?? null,
        plane: nameMatch ? Number(nameMatch[2]) : null,
        rankNumber,
        data: routerData,
    };
}

// Puts each chip in its mesh's grid by mesh_coord, row by row.
function placeChips(mesh, problems) {
    mesh.chipGrid = Array.from({ length: mesh.rows }, () => new Array(mesh.columns).fill(null));
    for (const chip of mesh.chips) {
        if (chip.row < 0 || chip.row >= mesh.rows || chip.column < 0 || chip.column >= mesh.columns) {
            problems.push(`${chip.path} has mesh_coord [${chip.row}, ${chip.column}], outside its ${mesh.rows}×${mesh.columns} mesh.`);
        } else if (mesh.chipGrid[chip.row][chip.column]) {
            problems.push(`${chip.path} and ${mesh.chipGrid[chip.row][chip.column].path} share mesh_coord [${chip.row}, ${chip.column}].`);
        } else {
            mesh.chipGrid[chip.row][chip.column] = chip;
        }
    }
    mesh.chips.sort((first, second) => first.row - second.row || first.column - second.column);
}

function groupRoutersByDirection(chip) {
    chip.routers.sort((first, second) =>
        directionRank(first.direction) - directionRank(second.direction) ||
        (first.plane ?? 0) - (second.plane ?? 0) ||
        first.name.localeCompare(second.name));
    for (const router of chip.routers) {
        const direction = router.direction ?? "?";
        if (!chip.routersByDirection.has(direction)) chip.routersByDirection.set(direction, []);
        chip.routersByDirection.get(direction).push(router);
    }
}

function directionRank(direction) {
    const index = DIRECTION_ORDER.indexOf(direction);
    return index === -1 ? DIRECTION_ORDER.length : index;
}

// One link per pair of routers, from link.peer. `check` says what both ends agree on:
//   both_ends        each end names the other as its peer
//   ends_disagree    the peer is loaded and names a different router, or none
//   peer_not_loaded  the peer is on a rank that isn't loaded
function buildLinks(routersByPath) {
    const links = [];
    const linkByRouterPath = new Map();
    for (const router of routersByPath.values()) {
        const peerPath = router.data.link?.peer;
        if (!peerPath || linkByRouterPath.has(router.path)) continue;
        const peer = routersByPath.get(peerPath) ?? null;
        let check = "both_ends";
        if (!peer) check = "peer_not_loaded";
        else if (peer.data.link?.peer !== router.path) check = "ends_disagree";
        const link = {
            ends: [router, peer],
            endPaths: [router.path, peerPath],
            wrap: Boolean(router.data.link.wrap),
            // The manifest's cross_host is true when the peer chip belongs to another MPI rank, which may run on the
            // same host (PhysicalSystemDescriptor::is_cross_host_eth_link compares ranks).
            crossRank: Boolean(router.data.link.cross_host),
            check,
        };
        links.push(link);
        linkByRouterPath.set(router.path, link);
        if (check === "both_ends") linkByRouterPath.set(peerPath, link);
    }
    return { links, linkByRouterPath };
}

// Every rank's discovery_skipped_cores, each with the router at that core when there is one. UMD's discovery skips a
// core whose Ethernet heartbeat doesn't advance; the capture still reads a router on it, so it can be complete.
function findDiscoverySkippedRouters(ranks, chipsByPath) {
    const routersByCore = new Map();
    for (const chip of chipsByPath.values()) {
        for (const router of chip.routers) {
            const [x, y] = router.data.identity?.virtual_core ?? [];
            routersByCore.set(`${chip.data.asic_id}/${x},${y}`, router);
        }
    }
    return ranks.flatMap((rank) => (rank.decoded.capture?.discovery_skipped_cores ?? []).map((skipped) => ({
        ...skipped,
        rankNumber: rank.number,
        router: routersByCore.get(`${skipped.asic_id}/${skipped.core.x},${skipped.core.y}`) ?? null,
    })));
}

function sumCoverage(ranks) {
    const total = {};
    for (const rank of ranks) {
        for (const [name, value] of Object.entries(rank.decoded.coverage ?? {})) {
            if (typeof value === "number") total[name] = (total[name] ?? 0) + value;
        }
    }
    return total;
}
