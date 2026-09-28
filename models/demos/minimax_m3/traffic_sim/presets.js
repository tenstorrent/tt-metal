// Named configurations. "today" presets mirror what runs on hardware (#57827); the rest add features.
// Filled in / tuned by sweep.js results (see README.md "Presets").
(function (root) {
  'use strict';
  const B_SPLIT = [1, 1, 1, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4];
  const P = [
    { name: 'today-A', desc: '4 gx, 16x[2,4], chunk 5120, even split, static 1M slots (#57827 baseline A)',
      cfg: { galaxies: 4, stages: 16, mesh: [2, 4], split: 'even', chunk: 5120, cache: 'slots' } },
    { name: 'today-C', desc: '4 gx, 16x[2,4], chunk 2048, split 1,1,1,5x5,4x8, static 1M slots (#57827 run C, current best)',
      cfg: { galaxies: 4, stages: 16, mesh: [2, 4], split: B_SPLIT, chunk: 2048, cache: 'slots' } },
  ];
  const API = { list: P, byName: (n) => { const p = P.find((x) => x.name === n); if (!p) throw new Error('unknown preset ' + n); return p; } };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.M3PRESETS = API;
})(typeof self !== 'undefined' ? self : this);
