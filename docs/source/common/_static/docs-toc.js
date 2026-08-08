(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    buildPageToc();
  });

  function buildPageToc() {
    // Skip the home/landing page
    if (document.querySelector('.hero')) return;

    var content = document.querySelector('.rst-content .document') ||
                  document.querySelector('.rst-content');
    if (!content) return;

    var headings = content.querySelectorAll('h2, h3');
    if (headings.length < 2) return;

    var toc = document.createElement('nav');
    toc.className = 'tt-page-toc';
    toc.setAttribute('aria-label', 'Page contents');

    var ul = document.createElement('ul');

    headings.forEach(function (h) {
      if (!h.id) {
        h.id = h.textContent.trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-|-$/g, '');
      }

      var li = document.createElement('li');
      li.className = h.tagName === 'H3' ? 'toc-h3' : 'toc-h2';

      var a = document.createElement('a');
      a.href = '#' + h.id;

      /* Strip headerlink anchors (¶ / [] symbols) before reading text */
      var clone = h.cloneNode(true);
      clone.querySelectorAll('a.headerlink').forEach(function (el) { el.remove(); });
      a.textContent = clone.textContent.trim();

      a.addEventListener('click', function (e) {
        e.preventDefault();
        h.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', '#' + h.id);
        setActive(a);
      });

      li.appendChild(a);
      ul.appendChild(li);
    });

    toc.appendChild(ul);
    document.body.appendChild(toc);

    setupScrollSpy(headings, ul);
  }

  function setActive(activeLink) {
    var toc = document.querySelector('.tt-page-toc');
    if (!toc) return;
    toc.querySelectorAll('a').forEach(function (a) {
      a.classList.remove('active');
    });
    activeLink.classList.add('active');
  }

  function setupScrollSpy(headings, ul) {
    var links = ul.querySelectorAll('a');
    var navbarHeight = parseInt(
      getComputedStyle(document.documentElement)
        .getPropertyValue('--tt-navbar-height') || '72', 10
    );

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var id = entry.target.id;
          links.forEach(function (a) {
            a.classList.toggle('active', a.getAttribute('href') === '#' + id);
          });
        }
      });
    }, {
      rootMargin: '-' + (navbarHeight + 8) + 'px 0px -70% 0px',
      threshold: 0
    });

    headings.forEach(function (h) {
      observer.observe(h);
    });

    if (links.length > 0) {
      links[0].classList.add('active');
    }
  }
})();
