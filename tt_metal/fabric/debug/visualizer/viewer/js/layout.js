// Chip-grid math is mirrored in viewer/tests/test_layout.py. Keep the constants and
// chipPosition / wrapControl formulas in lockstep.

export const CHIP_W = 72;
export const CHIP_H = 72;
export const CHIP_GAP = 28;
export const PORT = 10;
export const MESH_GAP = 96;
export const WRAP_BOW = 36;
export const PORT_GAP = 4;
export const PAD = 24;

function chipKey(meshId, chipId) {
  return `${meshId}:${chipId}`;
}

export function chipPosition(coord) {
  const y = coord[0];
  const x = coord[1];
  return {
    x: x * (CHIP_W + CHIP_GAP),
    y: y * (CHIP_H + CHIP_GAP),
  };
}

export function portSide(direction) {
  if (direction === "N") {
    return "top";
  }
  if (direction === "S") {
    return "bottom";
  }
  if (direction === "E") {
    return "right";
  }
  if (direction === "W") {
    return "left";
  }
  return "inside";
}

function packAlong(count, start, span) {
  if (count <= 1) {
    return [start + span / 2];
  }
  const step = Math.max(PORT + PORT_GAP, span / (count - 1));
  const used = step * (count - 1);
  const origin = start + Math.max(0, (span - used) / 2);
  return Array.from({ length: count }, (_, index) => origin + index * step);
}

function portPoint(chip, side, index, count) {
  const inset = PORT;
  if (side === "top") {
    return { x: packAlong(count, chip.x + inset, CHIP_W - inset * 2)[index], y: chip.y };
  }
  if (side === "bottom") {
    return { x: packAlong(count, chip.x + inset, CHIP_W - inset * 2)[index], y: chip.y + CHIP_H };
  }
  if (side === "left") {
    return { x: chip.x, y: packAlong(count, chip.y + inset, CHIP_H - inset * 2)[index] };
  }
  if (side === "right") {
    return { x: chip.x + CHIP_W, y: packAlong(count, chip.y + inset, CHIP_H - inset * 2)[index] };
  }
  return {
    x: chip.x + CHIP_W / 2 + (index - (count - 1) / 2) * (PORT + PORT_GAP),
    y: chip.y + CHIP_H / 2,
  };
}

export function wrapControl(x1, y1, x2, y2, meshBounds, axis) {
  if (axis === "ew") {
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const center = (meshBounds.minX + meshBounds.maxX) / 2;
    const cx = midX < center ? meshBounds.minX - WRAP_BOW : meshBounds.maxX + WRAP_BOW;
    return { cx, cy: midY };
  }
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const center = (meshBounds.minY + meshBounds.maxY) / 2;
  const cy = midY < center ? meshBounds.minY - WRAP_BOW : meshBounds.maxY + WRAP_BOW;
  return { cx: midX, cy };
}

export function wrapAxis(direction) {
  return direction === "N" || direction === "S" ? "ns" : "ew";
}

function meshChipBounds(chips) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const chip of chips) {
    minX = Math.min(minX, chip.x);
    minY = Math.min(minY, chip.y);
    maxX = Math.max(maxX, chip.x + CHIP_W);
    maxY = Math.max(maxY, chip.y + CHIP_H);
  }
  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: CHIP_W, maxY: CHIP_H };
  }
  return { minX, minY, maxX, maxY };
}

export function layoutModel(model) {
  const chips = [];
  const meshes = [];
  let originX = PAD;
  const originY = PAD;

  const meshesIn = [...model.decoded.topology.meshes].sort((left, right) => left.mesh_id - right.mesh_id);
  for (const mesh of meshesIn) {
    const placed = [];
    const unplaced = [];
    for (const chip of mesh.chips || []) {
      if (Array.isArray(chip.mesh_coord) && chip.mesh_coord.length === 2) {
        placed.push(chip);
      } else {
        unplaced.push(chip);
      }
    }

    const local = [];
    for (const chip of placed) {
      const position = chipPosition(chip.mesh_coord);
      local.push({
        key: chipKey(mesh.mesh_id, chip.fabric_chip_id),
        meshId: mesh.mesh_id,
        chipId: chip.fabric_chip_id,
        x: position.x,
        y: position.y,
        w: CHIP_W,
        h: CHIP_H,
        unplaced: false,
        chip,
      });
    }

    const gridBounds = meshChipBounds(local);
    let parkY = local.length ? gridBounds.maxY + CHIP_GAP : 0;
    unplaced.forEach((chip, index) => {
      local.push({
        key: chipKey(mesh.mesh_id, chip.fabric_chip_id),
        meshId: mesh.mesh_id,
        chipId: chip.fabric_chip_id,
        x: index * (CHIP_W + CHIP_GAP),
        y: parkY,
        w: CHIP_W,
        h: CHIP_H,
        unplaced: true,
        chip,
      });
    });

    const bounds = meshChipBounds(local);
    for (const chip of local) {
      chips.push({
        ...chip,
        x: originX + chip.x - bounds.minX,
        y: originY + chip.y - bounds.minY,
      });
    }
    const world = {
      meshId: mesh.mesh_id,
      x: originX,
      y: originY,
      w: bounds.maxX - bounds.minX,
      h: bounds.maxY - bounds.minY,
      minX: originX,
      minY: originY,
      maxX: originX + (bounds.maxX - bounds.minX),
      maxY: originY + (bounds.maxY - bounds.minY),
    };
    meshes.push(world);
    originX += world.w + MESH_GAP;
  }

  const chipsByKey = new Map(chips.map((chip) => [chip.key, chip]));
  const ports = [];
  const byChipSide = new Map();
  for (const router of model.routers) {
    const chip = model.chipOf.get(`${router.id.mesh_id}:${router.id.chip_id}:${router.id.eth_chan}`);
    if (!chip) {
      continue;
    }
    const key = chipKey(chip.mesh_id, chip.fabric_chip_id);
    const side = portSide(router.direction);
    const groupKey = `${key}:${side}`;
    if (!byChipSide.has(groupKey)) {
      byChipSide.set(groupKey, []);
    }
    byChipSide.get(groupKey).push(router);
  }
  for (const [groupKey, routers] of byChipSide) {
    const parts = groupKey.split(":");
    const side = parts.pop();
    const chip = chipsByKey.get(parts.join(":"));
    if (!chip) {
      continue;
    }
    routers
      .sort((left, right) => (left.routing_plane ?? 0) - (right.routing_plane ?? 0) || left.id.eth_chan - right.id.eth_chan)
      .forEach((router, index) => {
        const point = portPoint(chip, side, index, routers.length);
        ports.push({
          key: `${router.id.mesh_id}:${router.id.chip_id}:${router.id.eth_chan}`,
          chipKey: chip.key,
          side,
          x: point.x,
          y: point.y,
          router,
        });
      });
  }
  const portsByKey = new Map(ports.map((port) => [port.key, port]));

  const links = model.links.map((link, index) => {
    const srcKey = `${link.src.mesh_id}:${link.src.chip_id}:${link.src.eth_chan}`;
    const dstKey = link.dst ? `${link.dst.mesh_id}:${link.dst.chip_id}:${link.dst.eth_chan}` : null;
    const src = portsByKey.get(srcKey);
    const dst = dstKey ? portsByKey.get(dstKey) : null;
    const x1 = src ? src.x : 0;
    const y1 = src ? src.y : 0;
    let x2 = dst ? dst.x : x1 + 28;
    let y2 = dst ? dst.y : y1;
    if (!dst && src) {
      const side = src.side;
      x2 = x1 + (side === "left" ? -28 : side === "right" ? 28 : 0);
      y2 = y1 + (side === "top" ? -28 : side === "bottom" ? 28 : 0);
    }
    const mesh = meshes.find((entry) => entry.meshId === link.src.mesh_id);
    const wrap = Boolean(link.wrap) && dst;
    const axis = wrapAxis(link.direction);
    const control = wrap && mesh ? wrapControl(x1, y1, x2, y2, mesh, axis) : null;
    return {
      index,
      link,
      srcKey,
      dstKey,
      x1,
      y1,
      x2,
      y2,
      wrap,
      control,
    };
  });

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const include = (x, y) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  };
  for (const chip of chips) {
    include(chip.x, chip.y);
    include(chip.x + chip.w, chip.y + chip.h);
  }
  for (const port of ports) {
    include(port.x, port.y);
  }
  for (const link of links) {
    include(link.x1, link.y1);
    include(link.x2, link.y2);
    if (link.control) {
      include(link.control.cx, link.control.cy);
    }
  }
  if (!Number.isFinite(minX)) {
    minX = 0;
    minY = 0;
    maxX = 400;
    maxY = 300;
  }

  return {
    chips,
    ports,
    links,
    meshes,
    bounds: {
      minX: minX - PAD,
      minY: minY - PAD,
      maxX: maxX + PAD,
      maxY: maxY + PAD,
    },
  };
}
