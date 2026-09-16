export function endpointKey(endpoint) {
  if (!endpoint) {
    return null;
  }
  return `${endpoint.mesh_id}:${endpoint.chip_id}:${endpoint.eth_chan}`;
}

function chipKey(meshId, chipId) {
  return `${meshId}:${chipId}`;
}

export function endpointLabel(endpoint) {
  return `mesh ${endpoint.mesh_id} · chip ${endpoint.chip_id} · eth ${endpoint.eth_chan}`;
}

export function buildModel(decoded) {
  const byEndpoint = new Map();
  for (const router of decoded.routers) {
    const key = endpointKey(router.id);
    if (key === null || byEndpoint.has(key)) {
      throw new Error(
        key === null ? "Router is missing an endpoint" : `Duplicate router endpoint ${key}`,
      );
    }
    byEndpoint.set(key, router);
  }

  const chips = new Map();
  const chipOf = new Map();
  for (const mesh of decoded.topology.meshes) {
    for (const chip of mesh.chips || []) {
      const key = chipKey(mesh.mesh_id, chip.fabric_chip_id);
      chips.set(key, { ...chip, mesh_id: mesh.mesh_id });
      for (const endpoint of chip.routers || []) {
        chipOf.set(endpointKey(endpoint), chips.get(key));
      }
    }
  }

  const links = decoded.topology.links.map((link) => ({
    ...link,
    srcRouter: byEndpoint.get(endpointKey(link.src)) || null,
    dstRouter: link.dst ? byEndpoint.get(endpointKey(link.dst)) || null : null,
  }));

  const routers = [...byEndpoint.values()].sort((left, right) =>
    endpointKey(left.id).localeCompare(endpointKey(right.id), undefined, { numeric: true }),
  );

  return {
    decoded,
    routers,
    byEndpoint,
    chips,
    chipOf,
    links,
  };
}

export function routerSearchText(router) {
  const id = router.id || {};
  return [
    id.mesh_id,
    id.chip_id,
    id.eth_chan,
    `mesh ${id.mesh_id}`,
    `chip ${id.chip_id}`,
    `eth ${id.eth_chan}`,
    router.direction,
    router.link_class,
    router.capture?.status,
    router.lifecycle?.exit_state,
  ]
    .filter((value) => value !== null && value !== undefined)
    .join(" ")
    .toLowerCase();
}

export function filterRouters(model, query) {
  const terms = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  if (terms.length === 0) {
    return model.routers;
  }
  return model.routers.filter((router) => {
    const text = routerSearchText(router);
    return terms.every((term) => text.includes(term));
  });
}

export function provenanceHosts(decoded) {
  const hosts = new Set();
  for (const input of decoded.inputs || []) {
    const hostname = input.snapshot?.provenance?.hostname;
    if (hostname) {
      hosts.add(hostname);
    }
  }
  return [...hosts].sort();
}
